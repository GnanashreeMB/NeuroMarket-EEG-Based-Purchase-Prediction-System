"""
NeuroMarket - Interactive Simulator Dashboard
Control brain signals and see purchase prediction in real-time
"""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import numpy as np
import dash_bootstrap_components as dbc
import pandas as pd

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

# Product catalog - 30 products
products = [
    # Nike (4)
    {"id": 1, "name": "Nike Air Max", "brand": "Nike", "category": "Shoes", "price": 150},
    {"id": 2, "name": "Nike Dri-FIT Tee", "brand": "Nike", "category": "Apparel", "price": 35},
    {"id": 3, "name": "Nike Pro Shorts", "brand": "Nike", "category": "Apparel", "price": 40},
    {"id": 4, "name": "Nike Vaporfly", "brand": "Nike", "category": "Shoes", "price": 250},
    
    # Adidas (4)
    {"id": 5, "name": "Adidas Ultraboost", "brand": "Adidas", "category": "Shoes", "price": 180},
    {"id": 6, "name": "Adidas Originals Hoodie", "brand": "Adidas", "category": "Apparel", "price": 70},
    {"id": 7, "name": "Adidas Soccer Cleats", "brand": "Adidas", "category": "Shoes", "price": 120},
    {"id": 8, "name": "Adidas Track Pants", "brand": "Adidas", "category": "Apparel", "price": 55},
    
    # Apple (4)
    {"id": 9, "name": "iPhone 15", "brand": "Apple", "category": "Phone", "price": 999},
    {"id": 10, "name": "MacBook Pro", "brand": "Apple", "category": "Laptop", "price": 1299},
    {"id": 11, "name": "AirPods Pro", "brand": "Apple", "category": "Audio", "price": 249},
    {"id": 12, "name": "Apple Watch", "brand": "Apple", "category": "Wearable", "price": 399},
    
    # Samsung (4)
    {"id": 13, "name": "Galaxy S24", "brand": "Samsung", "category": "Phone", "price": 899},
    {"id": 14, "name": "Galaxy Tab", "brand": "Samsung", "category": "Tablet", "price": 649},
    {"id": 15, "name": "Galaxy Buds", "brand": "Samsung", "category": "Audio", "price": 149},
    {"id": 16, "name": "Galaxy Watch", "brand": "Samsung", "category": "Wearable", "price": 299},
    
    # Beverages (4)
    {"id": 17, "name": "Coca-Cola", "brand": "Coke", "category": "Beverage", "price": 2},
    {"id": 18, "name": "Pepsi", "brand": "Pepsi", "category": "Beverage", "price": 2},
    {"id": 19, "name": "Sprite", "brand": "Coke", "category": "Beverage", "price": 2},
    {"id": 20, "name": "Mountain Dew", "brand": "Pepsi", "category": "Beverage", "price": 2},
    
    # Fast Food (4)
    {"id": 21, "name": "McDonald's Burger", "brand": "McDonalds", "category": "Food", "price": 5},
    {"id": 22, "name": "Burger King Whopper", "brand": "Burger King", "category": "Food", "price": 5},
    {"id": 23, "name": "KFC Chicken", "brand": "KFC", "category": "Food", "price": 6},
    {"id": 24, "name": "Taco Bell Tacos", "brand": "Taco Bell", "category": "Food", "price": 3},
    
    # Cars (4)
    {"id": 25, "name": "Tesla Model 3", "brand": "Tesla", "category": "Car", "price": 45000},
    {"id": 26, "name": "BMW 3 Series", "brand": "BMW", "category": "Car", "price": 44000},
    {"id": 27, "name": "Mercedes C-Class", "brand": "Mercedes", "category": "Car", "price": 46000},
    {"id": 28, "name": "Audi A4", "brand": "Audi", "category": "Car", "price": 42000},
    
    # Coffee (2)
    {"id": 29, "name": "Starbucks Latte", "brand": "Starbucks", "category": "Coffee", "price": 5},
    {"id": 30, "name": "Dunkin Donuts Coffee", "brand": "Dunkin", "category": "Coffee", "price": 3},
]

# Simple prediction function based on brain signals
def predict_purchase(gamma, beta, theta, alpha, asymmetry):
    """
    Calculate purchase probability based on brain signals
    Gamma = emotional connection (0-10)
    Beta = engagement (0-10)
    Theta = memory (0-10)
    Alpha = relaxation (0-10)
    Asymmetry = approach/avoidance (-5 to +5)
    """
    
    # Base probability
    prob = 50
    
    # Gamma effect (emotional connection)
    if gamma > 7:
        prob += 25  # Strong emotional = more likely to buy
    elif gamma < 3:
        prob -= 20  # No emotion = less likely
    
    # Beta effect (engagement)
    if beta > 7:
        prob += 20  # Highly engaged
    elif beta < 3:
        prob -= 25  # Bored = won't buy
    
    # Theta effect (memory/nostalgia)
    if theta > 7:
        prob += 15  # Strong memories
    elif theta < 3:
        prob -= 5   # No connection
    
    # Alpha effect (relaxation)
    if alpha > 7:
        prob += 5   # Comfortable decision
    elif alpha < 3:
        prob -= 10  # Stressed = may not buy
    
    # Asymmetry effect (approach/avoidance)
    prob += asymmetry * 5  # Positive = want, Negative = avoid
    
    # Clip between 0-100
    prob = max(0, min(100, prob))
    
    # Determine confidence
    if prob > 80:
        confidence = "HIGH"
    elif prob > 60:
        confidence = "MEDIUM"
    elif prob > 40:
        confidence = "LOW"
    else:
        confidence = "VERY LOW"
    
    return prob, confidence

# App layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🧠 NEUROMARKET BRAIN SIMULATOR", 
                   className="text-center text-white mt-4 mb-2"),
            html.H4("Control brain signals to see how purchase confidence changes", 
                   className="text-center text-white-50 mb-4"),
        ])
    ]),
    
    # Main row
    dbc.Row([
        # Left column - Controls
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🎮 BRAIN SIGNAL CONTROLS", 
                              className="bg-primary text-white"),
                dbc.CardBody([
                    
                    html.Label("🧠 GAMMA (31-50 Hz) - Emotional Connection", className="text-white mt-2"),
                    dcc.Slider(
                        id='gamma-slider',
                        min=0, max=10, step=0.1, value=0,
                        marks={0: {'label': '0', 'style': {'color': 'white'}},
                               2: {'label': '2', 'style': {'color': 'white'}},
                               4: {'label': '4', 'style': {'color': 'white'}},
                               6: {'label': '6', 'style': {'color': 'white'}},
                               8: {'label': '8', 'style': {'color': 'white'}},
                               10: {'label': '10', 'style': {'color': 'white'}}},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Div("Low ← → High", className="text-white-50 small mb-3"),
                    
                    html.Label("👀 BETA (14-30 Hz) - Engagement/Attention", className="text-white mt-3"),
                    dcc.Slider(
                        id='beta-slider',
                        min=0, max=10, step=0.1, value=0,
                        marks={0: {'label': '0', 'style': {'color': 'white'}},
                               2: {'label': '2', 'style': {'color': 'white'}},
                               4: {'label': '4', 'style': {'color': 'white'}},
                               6: {'label': '6', 'style': {'color': 'white'}},
                               8: {'label': '8', 'style': {'color': 'white'}},
                               10: {'label': '10', 'style': {'color': 'white'}}},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Div("Bored ← → Engaged", className="text-white-50 small mb-3"),
                    
                    html.Label("💭 THETA (4-8 Hz) - Memory/Nostalgia", className="text-white mt-3"),
                    dcc.Slider(
                        id='theta-slider',
                        min=0, max=10, step=0.1, value=0,
                        marks={0: {'label': '0', 'style': {'color': 'white'}},
                               2: {'label': '2', 'style': {'color': 'white'}},
                               4: {'label': '4', 'style': {'color': 'white'}},
                               6: {'label': '6', 'style': {'color': 'white'}},
                               8: {'label': '8', 'style': {'color': 'white'}},
                               10: {'label': '10', 'style': {'color': 'white'}}},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Div("No recall ← → Strong memories", className="text-white-50 small mb-3"),
                    
                    html.Label("😌 ALPHA (8-14 Hz) - Relaxation/Comfort", className="text-white mt-3"),
                    dcc.Slider(
                        id='alpha-slider',
                        min=0, max=10, step=0.1, value=0,
                        marks={0: {'label': '0', 'style': {'color': 'white'}},
                               2: {'label': '2', 'style': {'color': 'white'}},
                               4: {'label': '4', 'style': {'color': 'white'}},
                               6: {'label': '6', 'style': {'color': 'white'}},
                               8: {'label': '8', 'style': {'color': 'white'}},
                               10: {'label': '10', 'style': {'color': 'white'}}},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Div("Stressed ← → Relaxed", className="text-white-50 small mb-3"),
                    
                    html.Label("⚖️ ASYMMETRY (-5 to +5) - Approach/Avoidance", className="text-white mt-3"),
                    dcc.Slider(
                        id='asymmetry-slider',
                        min=-5, max=5, step=0.1, value=0,
                        marks={-5: {'label': 'Avoid', 'style': {'color': 'white'}},
                               0: {'label': 'Neutral', 'style': {'color': 'white'}},
                               5: {'label': 'Approach', 'style': {'color': 'white'}}},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Div("Want to avoid ← → Want to approach", className="text-white-50 small mb-3"),
                    
                    html.Hr(className="bg-white"),
                    
                    html.Label("🛍️ SELECT PRODUCT", className="text-white"),
                    dcc.Dropdown(
                        id='product-dropdown',
                        options=[{'label': f"{p['name']} (${p['price']})", 'value': p['id']} 
                                for p in products],
                        placeholder="Select a product to begin...",
                        clearable=False,
                        style={'color': 'black', 'backgroundColor': 'white'}
                    ),
                    
                ])
            ])
        ], width=4),
        
        # Right column - Results
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🎯 PURCHASE PREDICTION", 
                                      className="bg-success text-white"),
                        dbc.CardBody([
                            html.Div(id='prediction-gauge', className="text-center"),
                            html.Div(id='confidence-text', className="text-center h3 mt-2 text-white"),
                        ])
                    ])
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 BRAIN SIGNAL ANALYSIS", 
                                      className="bg-info text-white"),
                        dbc.CardBody([
                            html.Div(id='signal-analysis', className="text-white")
                        ])
                    ])
                ], className="mt-3")
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("💡 MARKETING INSIGHT", 
                                      className="bg-warning text-dark"),
                        dbc.CardBody([
                            html.Div(id='marketing-insight', className="h5 text-dark")
                        ])
                    ])
                ], className="mt-3")
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🔍 WHY THIS PREDICTION?", 
                                      className="bg-danger text-white"),
                        dbc.CardBody([
                            html.Div(id='explanation-list', className="text-white")
                        ])
                    ])
                ], className="mt-3")
            ]),
            
            # Psychology Reference Section - WITH BLACK BACKGROUND AND BLACK TEXT
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(
                            html.Span("📚 BRAIN SIGNAL PSYCHOLOGY REFERENCE", 
                                     className="text-white"),
                            className="bg-secondary"
                        ),
                        dbc.CardBody([
                            dbc.Accordion([
                                # Gamma Accordion Item
                                dbc.AccordionItem([
                                    html.Div([
                                        html.H6("🧠 GAMMA (31-50 Hz) - Emotional Connection", 
                                               className="text-info"),
                                        html.Table([
                                            html.Thead(html.Tr([
                                                html.Th("Rating", className="text-dark"),
                                                html.Th("Meaning", className="text-dark"),
                                                html.Th("Psychology", className="text-dark")
                                            ])),
                                            html.Tbody([
                                                html.Tr([
                                                    html.Td("8-10", className="text-dark"),
                                                    html.Td("Very High", className="text-dark"),
                                                    html.Td("🔥 BUY: Deep emotional bonding", className="text-dark")
                                                ]),
                                                html.Tr([
                                                    html.Td("6-7", className="text-dark"),
                                                    html.Td("High", className="text-dark"),
                                                    html.Td("✅ BUY: Positive emotional resonance", className="text-dark")
                                                ]),
                                                html.Tr([
                                                    html.Td("4-5", className="text-dark"),
                                                    html.Td("Moderate", className="text-dark"),
                                                    html.Td("⚪ Neutral: Some interest but no emotional pull", className="text-dark")
                                                ]),
                                                html.Tr([
                                                    html.Td("2-3", className="text-dark"),
                                                    html.Td("Weak", className="text-dark"),
                                                    html.Td("⚠️ NOT BUY: No emotional connection", className="text-dark")
                                                ]),
                                                html.Tr([
                                                    html.Td("0-1", className="text-dark"),
                                                    html.Td("Very Weak", className="text-dark"),
                                                    html.Td("❌ STRONG NOT BUY: Complete emotional disconnect", className="text-dark")
                                                ]),
                                            ])
                                        ], className="table table-light table-striped w-100", style={"backgroundColor": "#ffffff"}),
                                        html.Small("Source: Frontal gamma oscillations correlate with haptic preference (Jastrzębska et al., 2022)", 
                                                 className="text-secondary mt-2 d-block")
                                    ])
                                ], title="🧠 GAMMA - Emotional Connection", 
                                   style={"backgroundColor": "#000000", "color": "white"}),
                                
                                # Beta Accordion Item
                                dbc.AccordionItem([
                                    html.Div([
                                        html.H6("👀 BETA (14-30 Hz) - Engagement/Attention", 
                                               className="text-info"),
                                        html.Table([
                                            html.Thead(html.Tr([
                                                html.Th("Rating", className="text-dark"),
                                                html.Th("Meaning", className="text-dark"),
                                                html.Th("Psychology", className="text-dark")
                                            ])),
                                            html.Tbody([
                                                html.Tr([
                                                    html.Td("8-10", className="text-dark"),
                                                    html.Td("Very High", className="text-dark"),
                                                    html.Td("🔥 BUY: Deep cognitive processing", className="text-dark")
                                                ]),
                                                html.Tr([
                                                    html.Td("6-7", className="text-dark"),
                                                    html.Td("High", className="text-dark"),
                                                    html.Td("✅ BUY: Engaged attention", className="text-dark")
                                                ]),
                                                html.Tr([
                                                    html.Td("4-5", className="text-dark"),
                                                    html.Td("Moderate", className="text-dark"),
                                                    html.Td("⚪ Neutral: Casual viewing", className="text-dark")
                                                ]),
                                                html.Tr([
                                                    html.Td("2-3", className="text-dark"),
                                                    html.Td("Low", className="text-dark"),
                                                    html.Td("⚠️ NOT BUY: Bored, attention wandering", className="text-dark")
                                                ]),
                                                html.Tr([
                                                    html.Td("0-1", className="text-dark"),
                                                    html.Td("Very Low", className="text-dark"),
                                                    html.Td("❌ STRONG NOT BUY: Complete disengagement", className="text-dark")
                                                ]),
                                            ])
                                        ], className="table table-light table-striped w-100", style={"backgroundColor": "#ffffff"}),
                                        html.Small("Source: Beta oscillations (14-23 Hz) most informative for predicting willingness-to-pay (Boksem & Smidts, 2015)", 
                                                 className="text-secondary mt-2 d-block")
                                    ])
                                ], title="👀 BETA - Engagement/Attention", 
                                   style={"backgroundColor": "#000000", "color": "white"}),
                                
                                # Theta Accordion Item
                                dbc.AccordionItem([
                                    html.Div([
                                        html.H6("💭 THETA (4-8 Hz) - Memory/Nostalgia", 
                                               className="text-info"),
                                        html.Table([
                                            html.Thead(html.Tr([
                                                html.Th("Rating", className="text-dark"),
                                                html.Th("Meaning", className="text-dark"),
                                                html.Th("Psychology", className="text-dark")
                                            ])),
                                            html.Tbody([
                                                html.Tr([
                                                    html.Td("8-10", className="text-dark"),
                                                    html.Td("Very High", className="text-dark"),
                                                    html.Td("🔥 BUY: Strong memory recall, nostalgia", className="text-dark")
                                                ]),
                                                html.Tr([
                                                    html.Td("6-7", className="text-dark"),
                                                    html.Td("High", className="text-dark"),
                                                    html.Td("✅ BUY: Familiar feelings, positive memories", className="text-dark")
                                                ]),
                                                html.Tr([
                                                    html.Td("4-5", className="text-dark"),
                                                    html.Td("Moderate", className="text-dark"),
                                                    html.Td("⚪ Neutral: Some recognition", className="text-dark")
                                                ]),
                                                html.Tr([
                                                    html.Td("2-3", className="text-dark"),
                                                    html.Td("Low", className="text-dark"),
                                                    html.Td("⚠️ NOT BUY: No memory connection", className="text-dark")
                                                ]),
                                                html.Tr([
                                                    html.Td("0-1", className="text-dark"),
                                                    html.Td("Very Low", className="text-dark"),
                                                    html.Td("❌ STRONG NOT BUY: Complete novelty (can be good or bad)", className="text-dark")
                                                ]),
                                            ])
                                        ], className="table table-light table-striped w-100", style={"backgroundColor": "#ffffff"}),
                                        html.Small("Source: Frontal theta activity increases when brain detects unexpected deals (Telpaz et al., 2015)", 
                                                 className="text-secondary mt-2 d-block")
                                    ])
                                ], title="💭 THETA - Memory/Nostalgia", 
                                   style={"backgroundColor": "#000000", "color": "white"}),
                                
                                # Alpha Accordion Item
                                dbc.AccordionItem([
                                    html.Div([
                                        html.H6("😌 ALPHA (8-14 Hz) - Relaxation/Comfort", 
                                               className="text-info"),
                                        html.Table([
                                            html.Thead(html.Tr([
                                                html.Th("Rating", className="text-dark"),
                                                html.Th("Meaning", className="text-dark"),
                                                html.Th("Psychology", className="text-dark")
                                            ])),
                                            html.Tbody([
                                                html.Tr([
                                                    html.Td("8-10", className="text-dark"),
                                                    html.Td("Very High", className="text-dark"),
                                                    html.Td("🔥 BUY: Comfortable decision", className="text-dark")
                                                ]),
                                                html.Tr([
                                                    html.Td("6-7", className="text-dark"),
                                                    html.Td("High", className="text-dark"),
                                                    html.Td("✅ BUY: At ease with product", className="text-dark")
                                                ]),
                                                html.Tr([
                                                    html.Td("4-5", className="text-dark"),
                                                    html.Td("Moderate", className="text-dark"),
                                                    html.Td("⚪ Neutral: Neither stressed nor relaxed", className="text-dark")
                                                ]),
                                                html.Tr([
                                                    html.Td("2-3", className="text-dark"),
                                                    html.Td("Low", className="text-dark"),
                                                    html.Td("⚠️ NOT BUY: Slight stress, uncertainty", className="text-dark")
                                                ]),
                                                html.Tr([
                                                    html.Td("0-1", className="text-dark"),
                                                    html.Td("Very Low", className="text-dark"),
                                                    html.Td("❌ STRONG NOT BUY: High anxiety, cognitive overload", className="text-dark")
                                                ]),
                                            ])
                                        ], className="table table-light table-striped w-100", style={"backgroundColor": "#ffffff"}),
                                        html.Small("Source: Higher alpha in left posterior cingulate associated with high product comprehension (Jin et al., 2024)", 
                                                 className="text-secondary mt-2 d-block")
                                    ])
                                ], title="😌 ALPHA - Relaxation/Comfort", 
                                   style={"backgroundColor": "#000000", "color": "white"}),
                                
                                # Asymmetry Accordion Item
                                dbc.AccordionItem([
                                    html.Div([
                                        html.H6("⚖️ ASYMMETRY (-5 to +5) - Approach/Avoidance", 
                                               className="text-info"),
                                        html.Table([
                                            html.Thead(html.Tr([
                                                html.Th("Rating", className="text-dark"),
                                                html.Th("Meaning", className="text-dark"),
                                                html.Th("Psychology", className="text-dark")
                                            ])),
                                            html.Tbody([
                                                html.Tr([
                                                    html.Td("+4 to +5", className="text-dark"),
                                                    html.Td("Strong Approach", className="text-dark"),
                                                    html.Td("🔥 BUY: Strong 'want it' motivation", className="text-dark")
                                                ]),
                                                html.Tr([
                                                    html.Td("+2 to +3", className="text-dark"),
                                                    html.Td("Moderate Approach", className="text-dark"),
                                                    html.Td("✅ BUY: Positive inclination", className="text-dark")
                                                ]),
                                                html.Tr([
                                                    html.Td("-1 to +1", className="text-dark"),
                                                    html.Td("Neutral", className="text-dark"),
                                                    html.Td("⚪ Undecided, motivational conflict", className="text-dark")
                                                ]),
                                                html.Tr([
                                                    html.Td("-2 to -3", className="text-dark"),
                                                    html.Td("Moderate Avoidance", className="text-dark"),
                                                    html.Td("⚠️ NOT BUY: Hesitant, wary", className="text-dark")
                                                ]),
                                                html.Tr([
                                                    html.Td("-4 to -5", className="text-dark"),
                                                    html.Td("Strong Avoidance", className="text-dark"),
                                                    html.Td("❌ STRONG NOT BUY: Active rejection", className="text-dark")
                                                ]),
                                            ])
                                        ], className="table table-light table-striped w-100", style={"backgroundColor": "#ffffff"}),
                                        html.Small("Source: Left frontal activity (positive) = approach, Right frontal (negative) = avoidance (Pfabigan et al., 2023)", 
                                                 className="text-secondary mt-2 d-block")
                                    ])
                                ], title="⚖️ ASYMMETRY - Approach/Avoidance", 
                                   style={"backgroundColor": "#000000", "color": "white"}),
                            ], start_collapsed=True),
                        ], style={"backgroundColor": "#000000", "padding": "15px"})
                    ])
                ], className="mt-3")
            ])
        ], width=8)
    ]),
    
    # Update interval
    dcc.Interval(id='update-trigger', interval=100)
], fluid=True)


@app.callback(
    [Output('prediction-gauge', 'children'),
     Output('confidence-text', 'children'),
     Output('signal-analysis', 'children'),
     Output('marketing-insight', 'children'),
     Output('explanation-list', 'children')],
    [Input('gamma-slider', 'value'),
     Input('beta-slider', 'value'),
     Input('theta-slider', 'value'),
     Input('alpha-slider', 'value'),
     Input('asymmetry-slider', 'value'),
     Input('product-dropdown', 'value')]
)
def update_dashboard(gamma, beta, theta, alpha, asymmetry, product_id):
    
    # If no product selected, show empty/zero state
    if product_id is None:
        # Empty gauge
        empty_gauge = dcc.Graph(
            figure=go.Figure(go.Indicator(
                mode="gauge+number",
                value=0,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Select a product", 'font': {'color': 'white'}},
                number={'font': {'color': 'white', 'size': 50}, 'suffix': ''},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': 'white', 'tickfont': {'color': 'white'}},
                    'bar': {'color': 'gray'},
                    'steps': [
                        {'range': [0, 100], 'color': 'darkgray'}
                    ]
                }
            )).update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=250,
                margin=dict(l=20, r=20, t=50, b=20),
                font={'color': 'white'}
            ),
            config={'displayModeBar': False}
        )
        
        empty_text = html.Div([
            html.H3("No product selected", className="text-white-50"),
            html.H5("Choose a product from dropdown", className="text-white-50")
        ])
        
        empty_analysis = html.Div("Adjust sliders and select a product to see analysis", className="text-white-50 text-center mt-4")
        empty_insight = "Select a product to see marketing insight"
        empty_explanation = html.Ul([html.Li("Select a product to see explanation", className="text-white-50")])
        
        return empty_gauge, empty_text, empty_analysis, empty_insight, empty_explanation
    
    # Get product
    product = next(p for p in products if p['id'] == product_id)
    
    # Calculate prediction
    prob, confidence = predict_purchase(gamma, beta, theta, alpha, asymmetry)
    
    # Gauge chart
    gauge = dcc.Graph(
        figure=go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Purchase Probability %", 'font': {'color': 'white'}},
            number={'font': {'color': 'white', 'size': 50}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': 'white', 'tickfont': {'color': 'white'}},
                'bar': {'color': 'lightgreen' if prob > 60 else 'orange' if prob > 40 else 'red'},
                'steps': [
                    {'range': [0, 40], 'color': 'darkred'},
                    {'range': [40, 60], 'color': 'darkorange'},
                    {'range': [60, 100], 'color': 'darkgreen'}
                ],
                'threshold': {
                    'line': {'color': 'white', 'width': 4},
                    'thickness': 0.75,
                    'value': prob
                }
            }
        )).update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=250,
            margin=dict(l=20, r=20, t=50, b=20),
            font={'color': 'white'}
        ),
        config={'displayModeBar': False}
    )
    
    # Confidence text
    pred_text = f"PREDICTION: {'BUY' if prob > 50 else 'NOT BUY'}"
    conf_text = f"Confidence: {confidence} ({prob:.1f}%)"
    
    # Signal analysis bars with white text
    signal_analysis = html.Div([
        html.Div([
            html.Span(f"🧠 GAMMA (Emotion): {gamma:.1f}/10", className="text-white"),
            html.Div(style={
                'width': f'{(gamma/10)*100}%',
                'height': '20px',
                'backgroundColor': '#ff6b6b',
                'marginTop': '5px',
                'borderRadius': '5px'
            })
        ], className="mb-2"),
        
        html.Div([
            html.Span(f"👀 BETA (Engagement): {beta:.1f}/10", className="text-white"),
            html.Div(style={
                'width': f'{(beta/10)*100}%',
                'height': '20px',
                'backgroundColor': '#4ecdc4',
                'marginTop': '5px',
                'borderRadius': '5px'
            })
        ], className="mb-2"),
        
        html.Div([
            html.Span(f"💭 THETA (Memory): {theta:.1f}/10", className="text-white"),
            html.Div(style={
                'width': f'{(theta/10)*100}%',
                'height': '20px',
                'backgroundColor': '#ffe66d',
                'marginTop': '5px',
                'borderRadius': '5px'
            })
        ], className="mb-2"),
        
        html.Div([
            html.Span(f"😌 ALPHA (Relaxation): {alpha:.1f}/10", className="text-white"),
            html.Div(style={
                'width': f'{(alpha/10)*100}%',
                'height': '20px',
                'backgroundColor': '#95e1d3',
                'marginTop': '5px',
                'borderRadius': '5px'
            })
        ], className="mb-2"),
        
        html.Div([
            html.Span(f"⚖️ ASYMMETRY (Approach): {asymmetry:.1f}/5", className="text-white"),
            html.Div(style={
                'width': f'{((asymmetry+5)/10)*100}%',
                'height': '20px',
                'backgroundColor': '#a8e6cf',
                'marginTop': '5px',
                'borderRadius': '5px'
            })
        ], className="mb-2"),
    ])
    
    # Marketing insight with proper colors based on card
    if prob > 70:
        insight = f"🔥 Strong purchase intent for {product['name']}! Emotional connection (Gamma={gamma:.1f}) and engagement (Beta={beta:.1f}) are high."
    elif prob > 50:
        insight = f"📊 Mixed signals for {product['name']}. Consider improving emotional appeal."
    elif prob > 30:
        insight = f"⚠️ Low purchase probability for {product['name']}. Boredom detected (Beta={beta:.1f}) or avoidance (Asymmetry={asymmetry:.1f})."
    else:
        insight = f"❌ Strong rejection for {product['name']}. Consumer actively avoiding this product."
    
    # Explanation list with white text
    explanations = []
    
    if gamma > 7:
        explanations.append(html.Li("🧠 Strong Gamma: Emotional connection driving purchase", className="text-white"))
    elif gamma < 3:
        explanations.append(html.Li("🔴 Low Gamma: No emotional connection", className="text-white"))
    
    if beta > 7:
        explanations.append(html.Li("👀 High Beta: Highly engaged with product", className="text-white"))
    elif beta < 3:
        explanations.append(html.Li("💤 Low Beta: Bored/disengaged", className="text-white"))
    
    if theta > 7:
        explanations.append(html.Li("💭 High Theta: Nostalgia/memory recall", className="text-white"))
    
    if asymmetry > 3:
        explanations.append(html.Li("⬆️ Positive Asymmetry: Approach motivation (wants it)", className="text-white"))
    elif asymmetry < -3:
        explanations.append(html.Li("⬇️ Negative Asymmetry: Avoidance motivation (doesn't want it)", className="text-white"))
    
    if alpha < 3:
        explanations.append(html.Li("😰 Low Alpha: Stress/anxiety about purchase", className="text-white"))
    
    if not explanations:
        explanations.append(html.Li("Neural signals mixed - consumer undecided", className="text-white"))
    
    explanation_list = html.Ul(explanations, className="text-white")
    
    return gauge, html.Div([html.H3(pred_text, className="text-white"), html.H5(conf_text, className="text-white")]), signal_analysis, insight, explanation_list


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 NEUROMARKET SIMULATOR STARTING...")
    print("📊 Dashboard: http://localhost:8050")
    print("="*60 + "\n")
    app.run(debug=True, port=8050)