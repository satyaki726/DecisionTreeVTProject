import numpy as np
import pandas as pd
import dash
import dash_html_components as html
import dash_table
import dash_core_components as dcc
from dash.dependencies import Input, Output
from pandas import DataFrame
from sklearn.datasets import make_blobs
import plotly.graph_objs as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

external_stylesheets = [
    {
        'href': 'https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css', 'rel': 'stylesheet',
        'integrity': 'sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO',
        'crossorigin': 'anonymous'
    }
]

X, y = make_blobs(n_samples=500, centers=3, n_features=2, center_box=(-4.0, 4.0))
# scatter plot, dots colored by class value
df = pd.DataFrame(dict(X=X[:, 0], Y=X[:, 1], Label=y))

options = [
    {'label': 'Gini', 'value': 'gini'},
    {'label': 'Entropy', 'value': 'entropy'}
]

fig = px.scatter(df, x="X", y="Y", color="Label")

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(children=[
    html.Header([
        html.Div([
            html.H1('Campus X')
        ], className='row')
    ], id='main-header'),
    html.Nav([
        html.Div([
            html.Ul([
                html.Li([
                    html.A(children="Home",href="https://mywbut.com/index")
                ]),
                html.Li([
                    html.A(children="Workshop",href="https://mywbut.com/training-workshop/elements-of-engineering")
                ]),
                html.Li([
                    html.A(children="Training",href="https://mywbut.com/training-workshop/online-summer-training")
                ]),
                html.Li([
                    html.A(children="Ask Question",href="https://mywbut.com/answer")
                ])
            ])
        ], className='row')
    ], id='navbar'),
    html.Section([
        html.Div([
            html.H1('Decision Tree VT')
        ], className='row')
    ], id="showcase"),
    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.H1("Create Data"),
                        html.Div([
                            html.Button("Generate", id='picker', n_clicks=0),
                        ], className='card-body')
                    ], id='button', className='card')
                ])
            ], className='col-md-12')
        ], className='row'),
        html.Div([
            html.Div([
                html.Section([
                    html.Div([
                        html.Div([
                            dash_table.DataTable(
                                id='table',
                                columns=[{"name": i, "id": i} for i in df.columns],
                            )
                        ], className='card-body1')
                    ], className='card')],id="main-section"),
                html.Aside([html.H1("Definition"), html.H6(
                    "A decision tree is a decision support tool that uses a tree-like model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility. It is one way to display an algorithm that only contains conditional control statements.Decision trees are commonly used in operations research, specifically in decision analysis, to help identify a strategy most likely to reach a goal, but are also a popular tool in machine learning.")],
                           id='sidebar')
            ],className='col-md-12'),
        ],className='row'),
        html.Div([
            html.H1(" ")
        ],className='row'),
        html.Div([
            html.Section([
                html.Div([
                    html.Div([
                        html.Div([
                              dcc.Graph(id='scatter')
                        ], className='card-body2')
                    ], className='card')
                ], className='col-md-12')],id="main-sec2"),
                html.Aside([html.H2("Construction"), html.H6(
                    "A tree can be learned by splitting the source set into subsets based on an attribute value test. This process is repeated on each derived subset in a recursive manner called recursive partitioning. The recursion is completed when the subset at a node all has the same value of the target variable, or when splitting no longer adds value to the predictions. The construction of decision tree classifier does not require any domain knowledge or parameter setting, and therefore is appropriate for exploratory knowledge discovery. Decision trees can handle high dimensional data. In general decision tree classifier has good accuracy. Decision tree induction is a typical inductive approach to learn knowledge on classification.")],
                           id='sidebar2')
        ],className='row'),
        html.Div([
            html.Div([
                html.Div([
                    html.H1('Decision Boundary & Accuracy Visualization')
                ], id='ned', className='card')
            ], className='col-md-12')
        ], className='row'),
        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        dcc.Graph(id='dt1')
                    ], className='card-body')
                ], className='card')
            ], className='col-md-6'),
            html.Div([
                html.Div([
                    html.Div([
                        dcc.Graph(id='bar2')
                    ], className='card-body')
                ], className='card')
            ], className='col-md-6')
        ], className='row'),
        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.H1('Overfitting & Underfitting Using HyperParameter On Decision Tree')
                    ],id='ut', className='card')
                ], className='card-body')
            ], className='col-md-12')
        ], className='row'),
        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        dcc.Dropdown(id='picker2', options=options, value='gini',clearable=False),
                        dcc.Input(id='input-on-submit', type='number',value=1, min=1,max=100),
                        dcc.Graph(id='dt2')
                    ], className='card')
                ], className='card-body')
            ], className='col-md-12')
        ], className='row'),
        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.H1('Dependency of Accuracy On Overfitting & Underfitting')
                    ], id='nt', className='card')
                ], className='card-body')
            ], className='col-md-12')
        ], className='row'),
        html.Div([
            html.H6("Enter the depth",className='card bg-warning'),
            html.Div(
                html.Div([
                    dcc.Input(id='button9', type='number',min=1,max=100)
                    ,html.Div(id='output9',className='card bg-info')],className='col-md-12'),
                )
        ],className='row'),
        html.Div(className='JNU')
    ], className='container')
])


@app.callback(Output('scatter', 'figure'), [Input('picker', 'n_clicks')])
def displayClick(n_clicks):
    global K, L
    global df
    K, L = make_blobs(n_samples=500, centers=3, n_features=2, center_box=(-4.0, 4.0))
    df = pd.DataFrame(dict(X=K[:, 0], Y=K[:, 1], Label=L))
    return px.scatter(df, x="X", y="Y", color="Label")

@app.callback(Output('table', 'data'), [Input('picker', 'n_clicks')])
def update(n_clicks):
    return df.to_dict('records')

@app.callback(Output('dt1', 'figure'), [Input('picker', 'n_clicks')])
def boundary(n_clicks):
    df_copy = df
    X = df_copy.iloc[:, 1:3].values
    y = df_copy.iloc[:, 0].values
    sc = StandardScaler()
    X = sc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    a = np.arange(start=X_train[:, 0].min() - 1, stop=X_train[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X_train[:, 1].min() - 1, stop=X_train[:, 1].max() + 1, step=0.01)
    XX, YY = np.meshgrid(a, b)
    arr = np.array([XX.ravel(), YY.ravel()]).T
    labels = clf.predict(arr)
    fig = go.Figure(data=go.Contour(z=labels.reshape(XX.shape), colorbar=dict(
        title='Decision Boundary',  # title here
        titleside='bottom',
        titlefont=dict(
            size=14,
            family='Arial, sans-serif'))))
    return fig

@app.callback(Output('dt2', 'figure'), [Input('picker2', 'value'), Input('input-on-submit', 'value')])
def updatethe(value1, value2):
    df_copy = df
    X = df_copy.iloc[:, 1:3].values
    y = df_copy.iloc[:, 0].values
    sc = StandardScaler()
    X = sc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = DecisionTreeClassifier(criterion=value1,max_depth=value2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    a = np.arange(start=X_train[:, 0].min() - 1, stop=X_train[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X_train[:, 1].min() - 1, stop=X_train[:, 1].max() + 1, step=0.01)
    XX, YY = np.meshgrid(a, b)
    arr = np.array([XX.ravel(), YY.ravel()]).T
    labels = clf.predict(arr)
    fig = go.Figure(data=go.Contour(z=labels.reshape(XX.shape), colorbar=dict(
        title='Decision Boundary',  # title here
        titleside='bottom',
        titlefont=dict(
            size=14,
            family='Arial, sans-serif'))))
    return fig

@app.callback(Output('bar2', 'figure'), [Input('picker', 'n_clicks')])
def accuracy(n_clicks):
    def DT(X, y, k=25):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf = DecisionTreeClassifier(max_depth=k)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        training_score = accuracy_score(y_train, clf.predict(X_train))
        test_score = accuracy_score(y_test, y_pred)
        return training_score, test_score

    df_copy = df
    X = df_copy.iloc[:, 1:3].values
    y = df_copy.iloc[:, 0].values
    sc = StandardScaler()
    X = sc.fit_transform(X)
    train = []
    test = []
    error1 = 0
    error2 = 0
    x1 = 0
    x2 = 0
    depth1 = 0
    depth2 = 0
    for i in range(1, 25):
        r2train, r2test = DT(X, y, k=i)
        if (r2train > r2test):
            x1 = r2train - r2test
        else:
            x2 = r2test - r2train
        if (error1 < x1):
            error1 = x1
            depth1 = i
        if (error2 < x2):
            error2 = x2
            depth2 = i
        train.append(r2train)
        test.append(r2test)
    x = np.arange(24) + 1
    x2 = np.arange(start=1, stop=25, step=1)
    x3 = x2.tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x3, y=train, mode='lines', name='training'))
    fig.add_trace(go.Scatter(x=x3, y=test, mode='lines', name='testing'))
    return fig

@app.callback(Output('output9', 'children'), [Input('button9', 'value')])
def accuracytester(value):
   df_copy = df
   X = df_copy.iloc[:, 1:3].values
   y = df_copy.iloc[:, 0].values
   sc = StandardScaler()
   X = sc.fit_transform(X)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   clf = DecisionTreeClassifier(max_depth=value)
   clf.fit(X_train, y_train)
   y_pred=clf.predict(X_test)
   training_score = accuracy_score(y_train, clf.predict(X_train))
   test_score = accuracy_score(y_test, y_pred)
   if(training_score>test_score):
       if(training_score-test_score>0.1):
           msg="Chances of overfitting is high"
       else:
           msg="Chances of overfitting is low"
   else:
       if(test_score-training_score>0.1):
           msg="chances of underfitting is high"
       else:
           msg="chances of underfitting is low"
   return "Training Accuracy:\n",training_score,"\n\n\n\n\nTesting Accuracy:\n",test_score,"\n Message:\n",msg





if __name__ == "__main__":
    app.run_server(debug=True)
