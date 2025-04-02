import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import datetime
import calendar
import requests
import io
import warnings

warnings.filterwarnings('ignore')

# GitHub相关配置
github_owner = "你的GitHub用户名"  # 替换为您的GitHub用户名
github_repo = "物料投放分析"  # 替换为您的GitHub仓库名
github_branch = "main"  # GitHub分支，通常是main或master

# GitHub文件路径
github_files = {
    "material_data": "2025物料源数据.xlsx",
    "sales_data": "25物料源销售数据.xlsx",
    "material_price": "物料单价.xlsx"
}

# 本地文件路径（备用）
local_files = {
    "material_data": r"C:\Users\何晴雅\Desktop\2025物料源数据.xlsx",
    "sales_data": r"C:\Users\何晴雅\Desktop\25物料源销售数据.xlsx",
    "material_price": r"C:\Users\何晴雅\Desktop\物料单价.xlsx"
}


# 从GitHub加载文件
def load_from_github(file_path, github_owner, github_repo, github_branch):
    raw_url = f"https://raw.githubusercontent.com/{github_owner}/{github_repo}/{github_branch}/{file_path}"
    try:
        response = requests.get(raw_url)
        response.raise_for_status()  # 确保请求成功
        return pd.read_excel(io.BytesIO(response.content))
    except Exception as e:
        print(f"从GitHub加载文件失败: {e}")
        return None


# 数据加载与处理
def load_data(use_github=True):
    # 尝试从GitHub加载数据，如果失败则使用本地文件
    if use_github:
        try:
            # 加载物料源数据
            material_data = load_from_github(github_files["material_data"], github_owner, github_repo, github_branch)

            # 加载销售数据
            sales_data = load_from_github(github_files["sales_data"], github_owner, github_repo, github_branch)

            # 加载物料单价数据
            material_price = load_from_github(github_files["material_price"], github_owner, github_repo, github_branch)

            # 检查是否所有数据都加载成功
            if material_data is None or sales_data is None or material_price is None:
                raise Exception("部分数据加载失败，将使用本地文件")

        except Exception as e:
            print(f"从GitHub加载数据失败，将使用本地文件: {e}")
            use_github = False

    # 如果GitHub加载失败或者选择使用本地文件，则加载本地文件
    if not use_github:
        # 加载物料源数据
        material_data = pd.read_excel(local_files["material_data"])

        # 加载销售数据
        sales_data = pd.read_excel(local_files["sales_data"])

        # 加载物料单价数据
        material_price = pd.read_excel(local_files["material_price"])

    # 处理日期格式
    material_data['发运月份'] = pd.to_datetime(material_data['发运月份'])
    sales_data['发运月份'] = pd.to_datetime(sales_data['发运月份'])

    # 创建月份和年份列
    for df in [material_data, sales_data]:
        df['月份'] = df['发运月份'].dt.month
        df['年份'] = df['发运月份'].dt.year
        df['月份名'] = df['发运月份'].dt.strftime('%Y-%m')

    # 计算物料成本
    material_data = pd.merge(material_data, material_price[['物料代码', '单价（元）']],
                             left_on='产品代码', right_on='物料代码', how='left')

    # 填充缺失的物料单价为平均值
    mean_price = material_price['单价（元）'].mean()
    material_data['单价（元）'].fillna(mean_price, inplace=True)

    # 计算物料总成本
    material_data['物料成本'] = material_data['求和项:数量（箱）'] * material_data['单价（元）']

    # 计算销售总金额
    sales_data['销售金额'] = sales_data['求和项:数量（箱）'] * sales_data['求和项:单价（箱）']

    # 按经销商、月份计算物料成本总和
    material_cost_by_distributor = material_data.groupby(['客户代码', '经销商名称', '月份名'])[
        '物料成本'].sum().reset_index()
    material_cost_by_distributor.rename(columns={'物料成本': '物料总成本'}, inplace=True)

    # 按经销商、月份计算销售总额
    sales_by_distributor = sales_data.groupby(['客户代码', '经销商名称', '月份名'])['销售金额'].sum().reset_index()
    sales_by_distributor.rename(columns={'销售金额': '销售总额'}, inplace=True)

    # 合并物料成本和销售数据
    distributor_data = pd.merge(material_cost_by_distributor, sales_by_distributor,
                                on=['客户代码', '经销商名称', '月份名'], how='outer').fillna(0)

    # 计算ROI
    distributor_data['ROI'] = np.where(distributor_data['物料总成本'] > 0,
                                       distributor_data['销售总额'] / distributor_data['物料总成本'], 0)

    # 计算物料销售比率
    distributor_data['物料销售比率'] = distributor_data['物料总成本'] / distributor_data['销售总额'].replace(0, np.nan)
    distributor_data['物料销售比率'].fillna(0, inplace=True)

    return material_data, sales_data, material_price, distributor_data


# 创建指标定义说明卡片
def create_metric_definition_card(title, definition):
    return dbc.Card(
        dbc.CardBody([
            html.H5(title, className="card-title"),
            html.P(definition, className="card-text")
        ]),
        className="mb-3"
    )


# 创建仪表盘
def create_dashboard(use_github=True):
    material_data, sales_data, material_price, distributor_data = load_data(use_github)

    # 初始化Dash应用
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP,
                                                    'https://use.fontawesome.com/releases/v5.15.1/css/all.css'])

    # 设置应用标题
    app.title = "物料投放分析动态仪表盘"

    # 区域列表
    regions = sorted(material_data['所属区域'].unique())

    # 省份列表
    provinces = sorted(material_data['省份'].unique())

    # 月份列表
    months = sorted(material_data['月份名'].unique())

    # 物料类别列表
    material_categories = sorted(material_price['物料类别'].unique())

    # 定义指标说明
    metric_definitions = {
        "物料总成本": "所有物料的成本总和，计算方式为：物料数量 × 物料单价。用于衡量物料投入的总金额。",
        "销售总额": "所有产品的销售收入总和，计算方式为：产品数量 × 产品单价。用于衡量销售业绩。",
        "投资回报率(ROI)": "投资回报率，计算方式为：销售总额 ÷ 物料总成本。ROI>1表示物料投入产生了正回报，ROI<1表示投入未获得有效回报。",
        "物料销售比率": "物料总成本占销售总额的百分比，计算方式为：物料总成本 ÷ 销售总额。该比率越低，表示物料使用效率越高。",
        "高效物料投放经销商": "ROI值较高的经销商，这些经销商能够高效地利用物料创造销售。",
        "待优化物料投放经销商": "ROI值较低的经销商，这些经销商的物料使用效率有待提高。"
    }

    # 创建仪表盘布局
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("物料投放分析动态仪表盘", className="text-center mb-4"),
                html.P("协助销售人员更好地利用物料以达到增长销售的目的", className="text-center mb-4")
            ])
        ]),

        dbc.Row([
            dbc.Col([
                html.H4("筛选条件", className="mt-2"),
                html.Label("选择区域:", className="font-weight-bold"),
                dcc.Dropdown(
                    id='region-dropdown',
                    options=[{'label': region, 'value': region} for region in regions],
                    value=regions,
                    multi=True
                ),
                html.Label("选择省份:", className="font-weight-bold mt-2"),
                dcc.Dropdown(
                    id='province-dropdown',
                    options=[{'label': province, 'value': province} for province in provinces],
                    value=provinces,
                    multi=True
                ),
                html.Label("选择月份:", className="font-weight-bold mt-2"),
                dcc.Dropdown(
                    id='month-dropdown',
                    options=[{'label': month, 'value': month} for month in months],
                    value=months[0],
                    multi=False
                ),
                html.Label("选择物料类别:", className="font-weight-bold mt-2"),
                dcc.Dropdown(
                    id='material-category-dropdown',
                    options=[{'label': category, 'value': category} for category in material_categories],
                    value=material_categories,
                    multi=True
                ),
                html.Br(),
                dbc.Button("更新仪表盘", id="update-button", color="primary", className="w-100 mt-2"),

                html.Hr(),
                html.H4("指标说明", className="mt-4"),
                html.Div([
                    create_metric_definition_card(key, value) for key, value in metric_definitions.items()
                ]),

                # 添加数据源说明
                html.Hr(),
                html.H4("数据来源", className="mt-4"),
                html.Div([
                    html.P("本仪表盘使用以下数据文件:"),
                    html.Ul([
                        html.Li("2025物料源数据.xlsx"),
                        html.Li("25物料源销售数据.xlsx"),
                        html.Li("物料单价.xlsx")
                    ]),
                    html.P("数据来源于GitHub仓库: " +
                           html.A(f"{github_owner}/{github_repo}",
                                  href=f"https://github.com/{github_owner}/{github_repo}",
                                  target="_blank"))
                ], className="mb-3")
            ], width=3),

            dbc.Col([
                dbc.Tabs([
                    dbc.Tab([
                        html.Div([
                            html.P("本页面显示关键业绩指标汇总和主要分析图表，帮助您快速了解物料投放效果。",
                                   className="text-muted mb-4")
                        ]),
                        html.Div(id='summary-stats'),
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id='roi-chart'),
                                html.Div([
                                    html.P(
                                        "此图表显示ROI最高的经销商，可以帮助您识别物料使用效率最高的经销商，向他们学习最佳实践。",
                                        className="text-muted")
                                ])
                            ], width=6),
                            dbc.Col([
                                dcc.Graph(id='top-materials-chart'),
                                html.Div([
                                    html.P("此图表显示发放数量最多的物料，帮助您了解哪些物料最受欢迎。",
                                           className="text-muted")
                                ])
                            ], width=6)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id='region-comparison-chart'),
                                html.Div([
                                    html.P(
                                        "此图表比较不同区域的物料成本、销售金额和ROI，帮助您识别表现最好和需要改进的区域。",
                                        className="text-muted")
                                ])
                            ], width=12)
                        ])
                    ], label="概览", tab_id="overview"),

                    dbc.Tab([
                        html.Div([
                            html.P("本页面分析物料投入和销售产出的关系，帮助您了解物料投入与销售业绩的相关性。",
                                   className="text-muted mb-4")
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id='material-sales-chart'),
                                html.Div([
                                    html.P(
                                        "此散点图展示物料成本与销售金额的关系，点的大小和颜色代表ROI。理想情况下，经销商应位于图表右上方（高销售额和高ROI）。",
                                        className="text-muted")
                                ])
                            ], width=12)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id='material-cost-trend'),
                                html.Div([
                                    html.P("此折线图显示物料成本的月度趋势，帮助您识别物料投入的季节性模式。",
                                           className="text-muted")
                                ])
                            ], width=6),
                            dbc.Col([
                                dcc.Graph(id='sales-trend'),
                                html.Div([
                                    html.P("此折线图显示销售金额的月度趋势，帮助您了解销售业绩的变化模式。",
                                           className="text-muted")
                                ])
                            ], width=6)
                        ])
                    ], label="物料与销售分析", tab_id="material-sales"),

                    dbc.Tab([
                        html.Div([
                            html.P("本页面分析各经销商的物料使用效率，帮助您识别表现优秀和需要改进的经销商。",
                                   className="text-muted mb-4")
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id='distributor-performance'),
                                html.Div([
                                    html.P(
                                        "此图表显示销售额最高的经销商的物料成本、销售金额和ROI，帮助您了解顶级经销商的表现。",
                                        className="text-muted")
                                ])
                            ], width=12)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.H4("高效物料投放经销商", className="text-center"),
                                html.P("以下经销商在物料使用上表现优异，ROI值较高，可作为标杆学习。",
                                       className="text-muted"),
                                html.Div(id='efficient-distributors-table')
                            ], width=6),
                            dbc.Col([
                                html.H4("待优化物料投放经销商", className="text-center"),
                                html.P("以下经销商在物料使用上有改进空间，ROI值较低，需要提供针对性指导。",
                                       className="text-muted"),
                                html.Div(id='inefficient-distributors-table')
                            ], width=6)
                        ])
                    ], label="经销商分析", tab_id="distributor"),

                    dbc.Tab([
                        html.Div([
                            html.P("本页面分析各物料的投资回报率，帮助您识别效果最佳和效果最差的物料类型。",
                                   className="text-muted mb-4")
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id='material-roi-chart'),
                                html.Div([
                                    html.P("此图表显示各物料的ROI值，帮助您识别投资回报率最高的物料。",
                                           className="text-muted")
                                ])
                            ], width=12)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.H4("物料投资回报分析", className="text-center"),
                                html.P("下表详细列出了各物料的成本、对应销售额和ROI值，帮助您评估各物料的投资回报效果。",
                                       className="text-muted"),
                                html.Div(id='material-roi-table')
                            ], width=12)
                        ])
                    ], label="ROI分析", tab_id="roi"),

                    dbc.Tab([
                        html.Div([
                            html.P("本页面基于数据分析结果，提供物料投放优化建议，帮助您提高物料使用效率和销售业绩。",
                                   className="text-muted mb-4")
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.H4("物料投放优化建议", className="text-center"),
                                html.Div(id='recommendations')
                            ], width=12)
                        ])
                    ], label="优化建议", tab_id="recommendations")
                ], id="tabs", active_tab="overview")
            ], width=9)
        ]),

        # 添加页脚，显示数据更新时间
        dbc.Row([
            dbc.Col([
                html.Hr(),
                html.P(f"数据最后更新时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                       className="text-muted text-center")
            ])
        ])
    ], fluid=True)

    # 定义回调函数 - 这部分保持不变
    @app.callback(
        [Output('summary-stats', 'children'),
         Output('roi-chart', 'figure'),
         Output('top-materials-chart', 'figure'),
         Output('region-comparison-chart', 'figure'),
         Output('material-sales-chart', 'figure'),
         Output('material-cost-trend', 'figure'),
         Output('sales-trend', 'figure'),
         Output('distributor-performance', 'figure'),
         Output('efficient-distributors-table', 'children'),
         Output('inefficient-distributors-table', 'children'),
         Output('material-roi-chart', 'figure'),
         Output('material-roi-table', 'children'),
         Output('recommendations', 'children')],
        [Input('update-button', 'n_clicks')],
        [State('region-dropdown', 'value'),
         State('province-dropdown', 'value'),
         State('month-dropdown', 'value'),
         State('material-category-dropdown', 'value')]
    )
    def update_dashboard(n_clicks, regions, provinces, month, material_categories):
        # 筛选数据
        filtered_material = material_data[
            (material_data['所属区域'].isin(regions)) &
            (material_data['省份'].isin(provinces)) &
            (material_data['月份名'] == month)
            ]

        # 合并物料类别信息到筛选后的物料数据
        filtered_material = pd.merge(filtered_material,
                                     material_price[['物料代码', '物料类别']],
                                     left_on='产品代码',
                                     right_on='物料代码',
                                     how='left')

        # 进一步筛选物料类别
        filtered_material = filtered_material[filtered_material['物料类别'].isin(material_categories)]

        # 筛选销售数据
        filtered_sales = sales_data[
            (sales_data['所属区域'].isin(regions)) &
            (sales_data['省份'].isin(provinces)) &
            (sales_data['月份名'] == month)
            ]

        # 筛选分销商数据
        filtered_distributor = distributor_data[
            distributor_data['月份名'] == month
            ]

        # 通过客户代码筛选分销商数据，确保只保留符合区域和省份条件的分销商
        valid_distributors = filtered_sales['客户代码'].unique()
        filtered_distributor = filtered_distributor[filtered_distributor['客户代码'].isin(valid_distributors)]

        # 创建汇总统计信息
        total_material_cost = filtered_material['物料成本'].sum()
        total_sales = filtered_sales['销售金额'].sum()
        roi = total_sales / total_material_cost if total_material_cost > 0 else 0
        total_distributors = filtered_sales['经销商名称'].nunique()

        # 添加更详细的卡片说明
        summary_stats = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{total_material_cost:.2f}元", className="card-title"),
                        html.P("物料总成本", className="card-text"),
                        html.Small("所有物料的成本总和", className="text-muted")
                    ])
                ], color="light", outline=True)
            ]),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{total_sales:.2f}元", className="card-title"),
                        html.P("销售总额", className="card-text"),
                        html.Small("所有产品的销售收入总和", className="text-muted")
                    ])
                ], color="light", outline=True)
            ]),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{roi:.2f}", className="card-title"),
                        html.P("投资回报率(ROI)", className="card-text"),
                        html.Small("销售总额 ÷ 物料总成本", className="text-muted")
                    ])
                ], color="light", outline=True)
            ]),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{total_distributors}", className="card-title"),
                        html.P("经销商数量", className="card-text"),
                        html.Small("筛选条件下的经销商总数", className="text-muted")
                    ])
                ], color="light", outline=True)
            ])
        ])

        # 创建ROI图表（增强版）
        roi_by_region = filtered_distributor.groupby('客户代码').agg({
            '物料总成本': 'sum',
            '销售总额': 'sum'
        }).reset_index()

        roi_by_region['ROI'] = roi_by_region['销售总额'] / roi_by_region['物料总成本'].replace(0, np.nan)
        roi_by_region['ROI'].fillna(0, inplace=True)

        # 合并经销商名称
        distributor_names = filtered_distributor[['客户代码', '经销商名称']].drop_duplicates()
        roi_by_region = pd.merge(roi_by_region, distributor_names, on='客户代码')

        # 按ROI排序并取前10名
        top_roi = roi_by_region.sort_values('ROI', ascending=False).head(10)

        # 添加气泡大小表示销售额
        roi_chart = px.bar(
            top_roi,
            x='经销商名称',
            y='ROI',
            title='<b>投资回报率(ROI)最高的经销商</b><br><sup>ROI = 销售总额 ÷ 物料总成本，ROI越高表示物料使用效率越高</sup>',
            color='ROI',
            color_continuous_scale='Viridis',
            text='ROI'
        )
        roi_chart.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        roi_chart.update_layout(
            xaxis_tickangle=-45,
            xaxis_title="经销商名称",
            yaxis_title="投资回报率(ROI)",
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            ),
            hovermode="closest"
        )

        # 创建热门物料图表（增强版）
        top_materials = filtered_material.groupby('产品名称')['求和项:数量（箱）'].sum().reset_index()
        top_materials = top_materials.sort_values('求和项:数量（箱）', ascending=False).head(10)

        top_materials_chart = px.bar(
            top_materials,
            x='产品名称',
            y='求和项:数量（箱）',
            title='<b>最热门物料 (按数量)</b><br><sup>显示发放数量最多的前10种物料</sup>',
            color='求和项:数量（箱）',
            color_continuous_scale='Blues',
            text='求和项:数量（箱）'
        )
        top_materials_chart.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        top_materials_chart.update_layout(
            xaxis_tickangle=-45,
            xaxis_title="物料名称",
            yaxis_title="数量（箱）",
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )

        # 创建区域比较图表（增强版）
        region_comparison = filtered_material.groupby('所属区域').agg({
            '物料成本': 'sum'
        }).reset_index()

        sales_by_region = filtered_sales.groupby('所属区域').agg({
            '销售金额': 'sum'
        }).reset_index()

        region_comparison = pd.merge(region_comparison, sales_by_region, on='所属区域', how='outer').fillna(0)
        region_comparison['ROI'] = region_comparison['销售金额'] / region_comparison['物料成本'].replace(0, np.nan)
        region_comparison['ROI'].fillna(0, inplace=True)

        region_comparison_chart = make_subplots(specs=[[{"secondary_y": True}]])

        region_comparison_chart.add_trace(
            go.Bar(x=region_comparison['所属区域'],
                   y=region_comparison['物料成本'],
                   name='物料成本',
                   marker_color='rgba(58, 71, 80, 0.6)',
                   hovertemplate='区域: %{x}<br>物料成本: %{y:.2f}元<extra></extra>'),
            secondary_y=False
        )

        region_comparison_chart.add_trace(
            go.Bar(x=region_comparison['所属区域'],
                   y=region_comparison['销售金额'],
                   name='销售金额',
                   marker_color='rgba(246, 78, 139, 0.6)',
                   hovertemplate='区域: %{x}<br>销售金额: %{y:.2f}元<extra></extra>'),
            secondary_y=False
        )

        region_comparison_chart.add_trace(
            go.Scatter(x=region_comparison['所属区域'],
                       y=region_comparison['ROI'],
                       name='ROI',
                       mode='lines+markers+text',
                       line=dict(color='rgb(25, 118, 210)', width=3),
                       marker=dict(size=10),
                       text=region_comparison['ROI'].round(2),
                       textposition='top center',
                       hovertemplate='区域: %{x}<br>ROI: %{y:.2f}<extra></extra>'),
            secondary_y=True
        )

        region_comparison_chart.update_layout(
            title_text='<b>区域比较: 物料成本、销售金额和ROI</b><br><sup>对比不同区域的物料投入和销售产出效果</sup>',
            barmode='group',
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        region_comparison_chart.update_yaxes(title_text='金额 (元)', secondary_y=False)
        region_comparison_chart.update_yaxes(title_text='ROI', secondary_y=True)

        # 创建物料-销售关系图（增强版）
        material_sales_relation = filtered_distributor.copy()

        material_sales_chart = px.scatter(
            material_sales_relation,
            x='物料总成本',
            y='销售总额',
            size='ROI',
            color='ROI',
            hover_name='经销商名称',
            log_x=True,
            log_y=True,
            title='<b>物料成本与销售金额关系</b><br><sup>散点图展示物料投入与销售产出的相关性，点的大小和颜色代表ROI值</sup>',
            color_continuous_scale='Viridis',
            size_max=40,
            hover_data={
                '物料总成本': ':.2f',
                '销售总额': ':.2f',
                'ROI': ':.2f'
            }
        )

        material_sales_chart.update_layout(
            xaxis_title="物料总成本 (元，对数刻度)",
            yaxis_title="销售总额 (元，对数刻度)",
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )

        # 添加参考线 - ROI=1
        material_sales_chart.add_trace(
            go.Scatter(
                x=[material_sales_relation['物料总成本'].min(), material_sales_relation['物料总成本'].max()],
                y=[material_sales_relation['物料总成本'].min(), material_sales_relation['物料总成本'].max()],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='ROI=1 参考线',
                showlegend=True
            )
        )

        # 创建物料成本趋势图（增强版）
        # 需要根据所有月份的数据创建趋势
        material_trend = material_data[
            (material_data['所属区域'].isin(regions)) &
            (material_data['省份'].isin(provinces))
            ]

        material_trend = material_trend.groupby('月份名')['物料成本'].sum().reset_index()

        material_cost_trend = px.line(
            material_trend,
            x='月份名',
            y='物料成本',
            title='<b>物料成本趋势</b><br><sup>显示各月份物料投入成本的变化</sup>',
            markers=True,
            line_shape='linear',
            render_mode='svg'
        )

        material_cost_trend.update_traces(
            marker=dict(size=10, symbol='circle', line=dict(width=2, color='DarkSlateGrey')),
            marker_color='rgb(0, 128, 255)',
            line=dict(width=3),
            hovertemplate='月份: %{x}<br>物料成本: %{y:.2f}元<extra></extra>'
        )

        material_cost_trend.update_layout(
            xaxis_title="月份",
            yaxis_title="物料成本 (元)",
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )

        # 创建销售趋势图（增强版）
        sales_trend_data = sales_data[
            (sales_data['所属区域'].isin(regions)) &
            (sales_data['省份'].isin(provinces))
            ]

        sales_trend_data = sales_trend_data.groupby('月份名')['销售金额'].sum().reset_index()

        sales_trend_chart = px.line(
            sales_trend_data,
            x='月份名',
            y='销售金额',
            title='<b>销售金额趋势</b><br><sup>显示各月份销售收入的变化</sup>',
            markers=True,
            line_shape='linear',
            render_mode='svg'
        )

        sales_trend_chart.update_traces(
            marker=dict(size=10, symbol='circle', line=dict(width=2, color='DarkSlateGrey')),
            marker_color='rgb(255, 64, 129)',
            line=dict(width=3),
            hovertemplate='月份: %{x}<br>销售金额: %{y:.2f}元<extra></extra>'
        )

        sales_trend_chart.update_layout(
            xaxis_title="月份",
            yaxis_title="销售金额 (元)",
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )

        # 创建经销商绩效图（增强版）
        distributor_performance = filtered_distributor.sort_values('销售总额', ascending=False).head(15)

        distributor_perf_chart = make_subplots(specs=[[{"secondary_y": True}]])

        distributor_perf_chart.add_trace(
            go.Bar(x=distributor_performance['经销商名称'],
                   y=distributor_performance['物料总成本'],
                   name='物料总成本',
                   marker_color='rgba(58, 71, 80, 0.6)',
                   hovertemplate='经销商: %{x}<br>物料总成本: %{y:.2f}元<extra></extra>'),
            secondary_y=False
        )

        distributor_perf_chart.add_trace(
            go.Bar(x=distributor_performance['经销商名称'],
                   y=distributor_performance['销售总额'],
                   name='销售总额',
                   marker_color='rgba(246, 78, 139, 0.6)',
                   hovertemplate='经销商: %{x}<br>销售总额: %{y:.2f}元<extra></extra>'),
            secondary_y=False
        )

        distributor_perf_chart.add_trace(
            go.Scatter(x=distributor_performance['经销商名称'],
                       y=distributor_performance['ROI'],
                       name='ROI',
                       mode='lines+markers+text',
                       line=dict(color='rgb(25, 118, 210)', width=3),
                       marker=dict(size=10),
                       text=distributor_performance['ROI'].round(2),
                       textposition='top center',
                       hovertemplate='经销商: %{x}<br>ROI: %{y:.2f}<extra></extra>'),
            secondary_y=True
        )

        distributor_perf_chart.update_layout(
            title_text='<b>顶级经销商绩效</b><br><sup>展示销售额前15的经销商的物料投入和销售产出情况</sup>',
            barmode='group',
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        distributor_perf_chart.update_xaxes(tickangle=-45)
        distributor_perf_chart.update_yaxes(title_text='金额 (元)', secondary_y=False)
        distributor_perf_chart.update_yaxes(title_text='ROI', secondary_y=True)

        # 创建高效经销商表格（增强版）
        efficient_distributors = filtered_distributor.sort_values('ROI', ascending=False).head(10)
        efficient_table = dbc.Table.from_dataframe(
            efficient_distributors[['经销商名称', '物料总成本', '销售总额', 'ROI']].round(2),
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
            className="table-sm table-hover"
        )

        # 创建低效经销商表格（增强版）
        # 只考虑物料成本大于0的经销商
        inefficient_distributors = filtered_distributor[filtered_distributor['物料总成本'] > 0].sort_values('ROI').head(
            10)
        inefficient_table = dbc.Table.from_dataframe(
            inefficient_distributors[['经销商名称', '物料总成本', '销售总额', 'ROI']].round(2),
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
            className="table-sm table-hover"
        )

        # 创建物料ROI图表（增强版）
        # 需要按物料类型计算ROI
        material_roi = filtered_material.groupby(['产品代码', '产品名称']).agg({
            '物料成本': 'sum',
            '求和项:数量（箱）': 'sum'
        }).reset_index()

        # 匹配销售数据
        product_codes = set(material_roi['产品代码'])

        # 假设产品代码的前6个字符与销售数据中的产品代码相匹配
        sales_by_product = filtered_sales.copy()
        sales_by_product['匹配代码'] = sales_by_product['产品代码'].str[:6]

        # 为物料数据创建匹配代码
        material_roi['匹配代码'] = material_roi['产品代码'].str[:6]

        # 按匹配代码汇总销售数据
        sales_summary = sales_by_product.groupby('匹配代码').agg({
            '销售金额': 'sum'
        }).reset_index()

        # 合并物料和销售数据
        material_roi = pd.merge(material_roi, sales_summary, on='匹配代码', how='left')
        material_roi['销售金额'].fillna(0, inplace=True)

        # 计算ROI
        material_roi['ROI'] = material_roi['销售金额'] / material_roi['物料成本'].replace(0, np.nan)
        material_roi['ROI'].fillna(0, inplace=True)

        # 只保留ROI > 0的物料
        material_roi = material_roi[material_roi['ROI'] > 0].sort_values('ROI', ascending=False)

        material_roi_chart = px.bar(
            material_roi.head(15),
            x='产品名称',
            y='ROI',
            title='<b>各物料ROI分析</b><br><sup>显示ROI最高的前15种物料，帮助识别最有效的物料类型</sup>',
            color='ROI',
            color_continuous_scale='Viridis',
            text='ROI'
        )
        material_roi_chart.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        material_roi_chart.update_layout(
            xaxis_tickangle=-45,
            xaxis_title="物料名称",
            yaxis_title="投资回报率(ROI)",
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )

        # 创建物料ROI分析表格（增强版）
        # 添加物料投入产出比率
        material_roi['物料投入产出比'] = (material_roi['物料成本'] / material_roi['销售金额'] * 100).round(2)
        material_roi_table = dbc.Table.from_dataframe(
            material_roi[['产品名称', '物料成本', '销售金额', 'ROI', '物料投入产出比']].round(2).rename(
                columns={'物料投入产出比': '物料投入产出比(%)'}, inplace=False),
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
            className="table-sm table-hover"
        )

        # 创建优化建议（增强版 - 添加更多图标和格式）
        # 1. 低ROI经销商优化
        low_roi_distributors = inefficient_distributors[['经销商名称', 'ROI', '物料总成本', '销售总额']].head(5)
        low_roi_recommendations = html.Div([
            html.H5([html.I(className="fas fa-exclamation-triangle text-warning mr-2"), "低ROI经销商优化建议:"]),
            html.Div([
                html.P("以下经销商的物料使用效率较低，ROI值小于1，表示物料投入未能获得相应的销售回报。建议采取以下措施:"),
                html.Ul([
                    html.Li([
                        html.Strong(f"{row['经销商名称']}"),
                        f": ROI仅为{row['ROI']:.2f}，物料成本{row['物料总成本']:.2f}元，销售额{row['销售总额']:.2f}元。",
                        html.Ul([
                            html.Li("分析物料使用方式，提供针对性培训"),
                            html.Li("调整物料投放结构，侧重高ROI物料"),
                            html.Li("减少总物料投入，提高使用效率")
                        ])
                    ]) for _, row in low_roi_distributors.iterrows()
                ])
            ], className="ml-4")
        ], className="mb-4 p-3 border rounded")

        # 2. 高潜力物料建议
        high_potential_materials = material_roi.head(5)
        material_recommendations = html.Div([
            html.H5([html.I(className="fas fa-lightbulb text-success mr-2"), "高潜力物料推广建议:"]),
            html.Div([
                html.P("以下物料表现出较高的ROI值，建议在未来的销售推广中重点使用:"),
                html.Ul([
                    html.Li([
                        html.Strong(f"{row['产品名称']}"),
                        f": ROI为{row['ROI']:.2f}，投入{row['物料成本']:.2f}元，带动销售{row['销售金额']:.2f}元。",
                        html.Br(),
                        html.Small(f"建议：增加投放量，扩大使用范围，作为重点推广物料。")
                    ]) for _, row in high_potential_materials.iterrows()
                ])
            ], className="ml-4")
        ], className="mb-4 p-3 border rounded")

        # 3. 区域策略建议
        region_strategy = region_comparison.sort_values('ROI', ascending=False)
        region_recommendations = html.Div([
            html.H5([html.I(className="fas fa-map-marked-alt text-primary mr-2"), "区域投放策略建议:"]),
            html.Div([
                html.P("不同区域的物料使用效率存在差异，建议根据ROI值调整区域物料投放策略:"),
                dbc.Table(
                    [
                        html.Thead(
                            html.Tr([
                                html.Th("区域"),
                                html.Th("ROI"),
                                html.Th("物料成本"),
                                html.Th("销售金额"),
                                html.Th("建议")
                            ])
                        ),
                        html.Tbody([
                            html.Tr([
                                html.Td(row['所属区域']),
                                html.Td(f"{row['ROI']:.2f}"),
                                html.Td(f"{row['物料成本']:.2f}元"),
                                html.Td(f"{row['销售金额']:.2f}元"),
                                html.Td(
                                    html.Span("加大投入", className="badge badge-success p-2")
                                    if row['ROI'] > 1 else
                                    html.Span("优化使用", className="badge badge-warning p-2")
                                )
                            ]) for _, row in region_strategy.iterrows()
                        ])
                    ],
                    bordered=True,
                    hover=True,
                    responsive=True,
                    striped=True,
                    className="table-sm"
                )
            ], className="ml-4")
        ], className="mb-4 p-3 border rounded")

        # 4. 投放时机建议
        timing_recommendations = html.Div([
            html.H5([html.I(className="fas fa-calendar-alt text-info mr-2"), "物料投放时机建议:"]),
            html.Div([
                html.P("根据销售趋势数据，为提高物料使用效率，建议:"),
                html.Ul([
                    html.Li("销售旺季前1-2个月增加物料投放量，确保销售人员有足够时间熟悉和使用物料"),
                    html.Li("在销售高峰期保持适量物料供应，确保及时补充"),
                    html.Li("销售淡季减少物料投放，避免库存积压"),
                    html.Li("季节性产品的物料应提前3个月准备，确保推广时效性")
                ])
            ], className="ml-4")
        ], className="mb-4 p-3 border rounded")

        # 5. 综合建议
        overall_recommendations = html.Div([
            html.H5([html.I(className="fas fa-cogs text-dark mr-2"), "综合优化建议:"]),
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H6("物料投放策略优化", className="mb-0")),
                            dbc.CardBody([
                                html.Ul([
                                    html.Li("针对不同区域和经销商定制化物料投放策略"),
                                    html.Li("根据历史ROI数据调整各类物料的投放量"),
                                    html.Li("重点推广高ROI物料，减少低ROI物料的投放")
                                ], className="mb-0")
                            ])
                        ], className="h-100")
                    ], width=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H6("物料使用管理改进", className="mb-0")),
                            dbc.CardBody([
                                html.Ul([
                                    html.Li("建立物料使用跟踪机制，定期评估物料使用效果"),
                                    html.Li("举办物料使用培训，提高销售人员对物料的有效利用"),
                                    html.Li("收集一线销售反馈，持续改进物料设计和投放策略")
                                ], className="mb-0")
                            ])
                        ], className="h-100")
                    ], width=6)
                ]),
                html.Br(),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H6("重点关注的经销商", className="mb-0")),
                            dbc.CardBody([
                                html.P("高潜力经销商 (销售额高但ROI低)：", className="font-weight-bold"),
                                html.Ul([
                                    html.Li(f"{row['经销商名称']} (ROI: {row['ROI']:.2f})")
                                    for _, row in filtered_distributor[
                                        (filtered_distributor['销售总额'] > filtered_distributor['销售总额'].median()) &
                                        (filtered_distributor['ROI'] < 1)
                                        ].sort_values('销售总额', ascending=False).head(3).iterrows()
                                ])
                            ])
                        ], className="h-100")
                    ], width=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H6("季度重点行动计划", className="mb-0")),
                            dbc.CardBody([
                                html.Ol([
                                    html.Li("对低ROI经销商进行物料使用诊断和培训"),
                                    html.Li("增加高ROI物料的投放比例，减少低效物料的使用"),
                                    html.Li("建立物料ROI评估机制，每月更新物料投放策略"),
                                    html.Li("实施物料与销售额挂钩的激励机制，提高物料利用率")
                                ])
                            ])
                        ], className="h-100")
                    ], width=6)
                ])
            ], className="ml-4")
        ], className="mb-4 p-3 border rounded")

        recommendations = html.Div([
            html.Div([
                html.P("以下是基于数据分析结果生成的物料投放优化建议，旨在提高物料使用效率和销售业绩。",
                       className="lead text-muted mb-4")
            ]),
            low_roi_recommendations,
            material_recommendations,
            region_recommendations,
            timing_recommendations,
            overall_recommendations
        ])

        return (
            summary_stats, roi_chart, top_materials_chart, region_comparison_chart,
            material_sales_chart, material_cost_trend, sales_trend_chart, distributor_perf_chart,
            efficient_table, inefficient_table, material_roi_chart, material_roi_table,
            recommendations
        )

    return app


# 部署到GitHub相关函数
def create_readme_content(github_owner, github_repo):
    """创建GitHub仓库的README.md文件内容"""
    return f"""# 物料投放分析动态仪表盘

## 项目简介
该项目是一个物料投放分析动态仪表盘，旨在协助销售人员更好地利用物料以达到增长销售的目的。

## 主要功能
1. 物料成本与销售额关系分析
2. 经销商ROI分析
3. 区域物料投放效率比较
4. 物料类型效果分析
5. 提供优化建议

## 使用方法
1. 下载代码
2. 安装所需Python库：pandas, numpy, plotly, dash, dash-bootstrap-components, requests
3. 运行Python脚本

## 数据文件
- 2025物料源数据.xlsx
- 25物料源销售数据.xlsx
- 物料单价.xlsx

## 在线访问
您可以通过以下链接访问该仪表盘：
https://{github_owner}.github.io/{github_repo}/

## 联系方式
如有问题，请联系：[您的联系方式]
"""


def create_deployment_instructions():
    """创建GitHub Pages部署说明"""
    instructions = """# 部署到GitHub Pages的步骤

1. 创建GitHub仓库并上传代码和数据文件
2. 启用GitHub Pages
   - 进入仓库设置
   - 找到"GitHub Pages"部分
   - 选择分支（通常是main或master）
   - 保存设置
3. 如需使用自定义域名：
   - 在仓库中创建CNAME文件
   - 在域名提供商处添加相应DNS记录

备注：由于GitHub Pages不支持直接运行Python代码，您可能需要使用以下方法之一：
- 部署到支持Python的平台（如Heroku, AWS, Azure等）
- 使用GitHub Actions自动构建静态站点
- 使用静态导出版本的仪表盘
"""
    return instructions


if __name__ == '__main__':
    # 默认使用GitHub数据源
    app = create_dashboard(use_github=True)
    app.run_server(debug=True, host='0.0.0.0', port=8050)