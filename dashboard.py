import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import io
import base64
import time
import webbrowser
from threading import Timer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from gurobi_optimizer import optimize_with_gurobi
from pulp_optimizer import optimize_with_pulp

app = dash.Dash(__name__)

#excel downlaod file template
def download_template():
    output = io.BytesIO()
    df = pd.DataFrame(columns=["release date", "due date", "weight", "st1", "st2", "st3"])
    df.to_excel(output, index=False, engine='openpyxl')
    output.seek(0)
    return output

#load data from excel file
def load_task_data(uploaded_file):
    df = pd.read_excel(uploaded_file, engine="openpyxl")
    
    tasks = df[['release date', 'due date', 'weight']].copy()
    tasks['id'] = range(1, len(tasks) + 1)

    service_time_columns = [col for col in df.columns if col.startswith('st')]
    tasks['service_times'] = df[service_time_columns].values.tolist()
    tasks['num_machines'] = len(service_time_columns)

    return tasks 

#gantt chart generation
def generate_gantt_chart(results, tasks, execution_time):

    gantt_data = []
    total_weighted_tardiness = 0  
    
    task_names = list(set([f"Task {task['id']}" for task in results['tasks']]))
    
    #color mapping
    cmap = plt.get_cmap("tab20")
    num_tasks = len(task_names)
    custom_colors = [matplotlib.colors.rgb2hex(cmap(i / num_tasks)) for i in range(num_tasks)]

    task_color_map = {task: custom_colors[i] for i, task in enumerate(task_names)}
            
    #data prep
    for task in results['tasks']:
        task_id = task["id"]
        start_times = task["start_times"]
        tardiness = task["tardiness"]
        service_times = tasks.loc[tasks['id'] == task_id, 'service_times'].values[0]
        weight = tasks.loc[tasks['id'] == task_id, 'weight'].values[0]
        
        total_weighted_tardiness += weight * tardiness

        for machine_idx, (start_time, service_time) in enumerate(zip(start_times, service_times)):
            task_name = f"Task {task_id}"
            gantt_data.append({
                "Task": task_name,
                "Machine": f"Machine {machine_idx + 1}",
                "Start": start_time,  
                "Finish": start_time + service_time,  
                "Tardiness": tardiness,
                "Weight": weight,
                "Duration": service_time,  
                "Color": task_color_map[task_name]  
            })
    
    df = pd.DataFrame(gantt_data)
    
    #use plotly to generate gantt chart
    fig = go.Figure()
    fig_data = []

    #bar generation
    for idx, row in df.iterrows():
        fig_data.append(go.Bar(
            x=[row['Duration']],
            y=[row['Machine']],
            orientation='h',
            base=row['Start'],
            name=row['Task'],
            marker=dict(color=row['Color']),
            hovertemplate=(
                f"Task: {row['Task']}<br>"
                f"Machine: {row['Machine']}<br>"
                f"Start: {row['Start']}<br>"
                f"Finish: {row['Finish']}<br>"
                f"Tardiness: {row['Tardiness']}<br>"
                f"Weight: {row['Weight']}<extra></extra>"
            )
        ))

    fig = go.Figure(data=fig_data)

    fig.add_annotation(
        x=0.5,  
        y=1.1,  
        text=f"Total Weighted Tardiness: {total_weighted_tardiness:.2f}",
        showarrow=False,
        font=dict(size=16, color="black"),
        align="center",
        xref="paper",  
        yref="paper", 
        bgcolor="#F5F5F5"  
    )

    #custom layout
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Machine",
        barmode='stack',  
        bargap=0.1, 
        height=500,
        showlegend=False,  
        hovermode="closest",  
        plot_bgcolor='#F5F5F5',
        xaxis=dict(
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=False
        )
    )


    return dcc.Graph(figure=fig)



def format_results_to_df(results, tasks):
    task_details = []

    for task in results['tasks']:
        task_id = task['id']
        start_times = task["start_times"]

        #get details
        task_data = tasks[tasks['id'] == task_id].iloc[0]
        release_date = task_data['release date']
        due_date = task_data['due date']
        weight = task_data['weight']
        service_times = task_data['service_times']

        #final date calc
        done_date = max(start_times[i] + service_times[i] for i in range(len(service_times)))

        #tardiness calc
        tardiness = max(0, (done_date - due_date) * weight)

        task_row = {
            'Task ID': task_id,
            'Release Date': round(release_date),
            'Due Date': round(due_date),
            'Weight': round(weight),
        }


        for machine_idx, (start_time, service_time) in enumerate(zip(start_times, service_times)):
            task_row[f"Start Time M{machine_idx + 1}"] = round(start_time)
            task_row[f"Finish Time M{machine_idx + 1}"] = round(start_time + service_time)


        task_row['Done Date'] = round(done_date)
        task_row['Tardiness'] = round(tardiness)

        task_details.append(task_row)


    df = pd.DataFrame(task_details)

    columns = [col for col in df.columns if col not in ['Done Date', 'Tardiness']]
    columns += ['Done Date', 'Tardiness'] #done date and tardiness at the end
    df = df[columns]

    return df

#check feasibility of schedule    
def check_feasibility(tasks, schedule_results):

    checks = {
        "Release dates respected": True,
        "No overlapping tasks on machines": True,
        "Due dates and tardiness correctly calculated": True,
    }

    #release date repected check
    for task in schedule_results["tasks"]:
        task_data = tasks[tasks["id"] == task["id"]].iloc[0]
        if task["start_times"][0] < task_data["release date"]:
            checks["Release dates respected"] = False
            break

    #no overlapping tasks on machines
    machine_schedules = {}
    for task in schedule_results["tasks"]:
        task_data = tasks[tasks["id"] == task["id"]].iloc[0]  
        for idx, start_time in enumerate(task["start_times"]):
            service_time = task_data["service_times"][idx]
            finish_time = start_time + service_time
            machine_schedules.setdefault(idx, []).append((start_time, finish_time))


    # Check for overlapping intervals
    for machine, intervals in machine_schedules.items():
        intervals.sort()
        for i in range(len(intervals) - 1):
            if intervals[i][1] > intervals[i + 1][0]:
                checks["No overlapping tasks on machines"] = False
                break

    # Check due dates and tardiness calculations
    for task in schedule_results["tasks"]:
        task_data = tasks[tasks["id"] == task["id"]].iloc[0]
        finish_time = task["finish_time"]
        due_date = task_data["due date"]
        tardiness = max(0, finish_time - due_date)
        if not abs(tardiness - task["tardiness"]) < 1e-5:
            checks["Due dates and tardiness correctly calculated"] = False
            break

    return checks



app.layout = html.Div([
    #tabs
    dcc.Tabs([
        #first tab
        dcc.Tab(label='Task Scheduling', children=[
            html.Div([
                #input data and constraints block
                html.Div([
                    html.H3("Input Data & Constraints", style={
                        'fontSize': '24px', 
                        'fontFamily': 'Arial, sans-serif', 
                        'marginBottom': '10px'
                    }),
                    #download template button
                    html.Button(
                        id="download-template-button",
                        children="Download Template",
                        style={
                            'width': '100%',
                            'height': '50px',
                            'lineHeight': '50px',
                            'borderWidth': '1px',
                            'borderStyle': 'solid',
                            'borderRadius': '10px',
                            'backgroundColor': '#17a2b8',
                            'color': 'white',
                            'fontSize': '18px',
                            'cursor': 'pointer',
                            'marginBottom': '20px',
                            'transition': 'background-color 0.3s',
                        }
                    ),
                    dcc.Download(id="download-template-file"),  
                    dcc.Upload(
                        id="upload-data",
                        children=html.Button("Upload Excel File", style={
                            'width': '100%',
                            'height': '50px',
                            'lineHeight': '50px',
                            'borderWidth': '1px',
                            'borderStyle': 'solid',
                            'borderRadius': '10px',
                            'backgroundColor': '#4CAF50',
                            'color': 'white',
                            'fontSize': '18px',
                            'cursor': 'pointer',
                            'transition': 'background-color 0.3s',
                        }),
                        style={
                            'width': '95%',
                            'borderWidth': '1px',
                            'borderRadius': '10px',
                            'borderStyle': 'dashed',
                            'textAlign': 'center',
                            'padding': '20px',
                            'marginBottom': '20px',
                            'boxShadow': '0px 4px 10px rgba(0,0,0,0.1)',
                            'backgroundColor': '#f9f9f9',
                        }
                    ),
                    html.Div(id="file-status", style={
                        'marginBottom': '20px', 
                        'fontSize': '16px', 
                        'color': '#28a745',
                        'fontFamily': 'Arial, sans-serif'
                    }),
                    html.Label("Time Limit (seconds):", style={
                        'fontSize': '18px', 
                        'fontFamily': 'Arial, sans-serif'
                    }),
                    dcc.Input(
                        id="time-limit",
                        type="number",
                        value=60,
                        min=1,
                        step=1,
                        style={
                            'width': '100%',
                            'padding': '12px',
                            'fontSize': '18px',
                            'borderRadius': '8px',
                            'border': '1px solid #ccc',
                            'marginBottom': '20px'
                        }
                    ),
                    html.Label("Algorithm Selection:", style={
                        'fontSize': '18px', 
                        'fontFamily': 'Arial, sans-serif'
                    }),
                    dcc.Dropdown(
                        id="algorithm-selection",
                        options=[
                            {"label": "Gurobi Algorithm (license needed!)", "value": "gurobi"},
                            {"label": "PuLP Algorithm", "value": "pulp"}
                        ],
                        value="gurobi",  
                        style={
                            'width': '100%',
                            'marginBottom': '20px'
                        }
                    ),
                    html.Label("Task Precedence Constraint:", style={
                        'fontSize': '18px', 
                        'fontFamily': 'Arial, sans-serif'
                    }),
                    dcc.Checklist(
                        id="precedence-check",
                        options=[{"label": "Enforce precedence across machines", "value": "precedence"}],
                        value=[],
                        style={
                            'marginBottom': '20px', 
                            'fontSize': '16px',
                            'fontFamily': 'Arial, sans-serif'
                        }
                    ),
                    html.Div(id="output-status", style={
                        'marginTop': '20px', 
                        'color': '#dc3545', 
                        'fontSize': '16px',
                        'fontFamily': 'Arial, sans-serif'
                    }),

                    
                    #optimization button
                    html.Button(
                        id='run-model-button',
                        n_clicks=0,
                        children='Optimize schedule',
                        style={
                            'width': '100%',
                            'height': '50px',
                            'lineHeight': '50px',
                            'borderRadius': '10px',
                            'background': 'blue',  
                            'color': 'white',
                            'fontSize': '20px',
                            'cursor': 'pointer',
                            'transition': 'background-color 0.3s',
                            'border': 'none',
                            'marginTop': '20px',
                            'boxShadow': '0px 4px 10px rgba(0,0,0,0.1)'
                        }
                    ),
                ], style={
                    'width': '30%',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'padding': '20px',
                    'borderRight': '2px solid #ccc',
                    'backgroundColor': '#f4f4f9',
                    'borderRadius': '8px',
                    'boxShadow': '0px 6px 15px rgba(0,0,0,0.1)',
                }),

                #plot block
                html.Div([
                    html.H3("Optimized schedule", style={
                        'fontSize': '24px', 
                        'fontFamily': 'Arial, sans-serif', 
                        'marginBottom': '20px'
                    }),
                    html.Div(id="input-data-preview", style={'textAlign': 'center', 'marginBottom': '20px'}),  
                    html.Div(id="output-graph", style={'textAlign': 'center'})
                ], style={
                    'width': 'calc(100% - 30%)',  
                    'display': 'inline-block',
                    'padding': '20px',
                    'textAlign': 'center',
                    'backgroundColor': '#ffffff',
                    'borderRadius': '8px',
                    'boxShadow': '0px 6px 15px rgba(0,0,0,0.1)'
                })
            ], style={'display': 'flex', 'justifyContent': 'space-between' }),

            #three blocks with stats
            html.Div([
                html.Div([
                    html.H5("Number of tasks late", style={
                        'textAlign': 'center', 
                        'color': 'black', 
                        'marginBottom': '10px', 
                        'marginTop': '1px',
                        'fontWeight': '500', 
                        'fontSize': '1.5em',
                        'fontFamily': 'Arial, sans-serif'
                    }),
                    html.Div(id="tasks-late", style={
                        'textAlign': 'center', 
                        'fontSize': '2.5em', 
                        'fontWeight': '700', 
                        'color': 'black',
                        'fontFamily': 'Arial, sans-serif'
                    })
                ], style={
                    'flex': '1',
                    'padding': '20px',
                    'borderRadius': '15px',
                    'background': 'linear-gradient(135deg, #e65758, #771d32)',  
                    'boxShadow': '0px 10px 20px rgba(138, 58, 185, 0.4)',  
                    'margin': '0 10px',
                }),
                html.Div([
                    html.H5("Number of tasks on time", style={
                        'textAlign': 'center', 
                        'color': 'black', 
                        'marginBottom': '10px', 
                        'marginTop': '1px',
                        'fontWeight': '500', 
                        'fontSize': '1.5em',
                        'fontFamily': 'Arial, sans-serif'
                    }),
                    html.Div(id="tasks-on-time", style={
                        'textAlign': 'center', 
                        'fontSize': '2.5em', 
                        'fontWeight': '700', 
                        'color': 'black',
                        'fontFamily': 'Arial, sans-serif'
                    })
                ], style={
                    'flex': '1',
                    'padding': '20px',
                    'borderRadius': '15px',
                    'background': 'linear-gradient(135deg, #0ccda3, #c1fcd3)',  
                    'boxShadow': '0px 10px 20px rgba(60, 165, 93, 0.4)', 
                    'margin': '0 10px',
                }),
                html.Div([
                    html.H5("Max Tardiness", style={
                        'textAlign': 'center', 
                        'color': 'black', 
                        'marginBottom': '10px', 
                        'marginTop': '1px',
                        'fontWeight': '500', 
                        'fontSize': '1.5em',
                        'fontFamily': 'Arial, sans-serif' 
                    }),
                    html.Div(id="max-tardiness", style={
                        'textAlign': 'center', 
                        'fontSize': '2.5em', 
                        'fontWeight': '700', 
                        'color': 'black',
                        'fontFamily': 'Arial, sans-serif'
                    })
                ], style={
                    'flex': '1',
                    'padding': '20px',
                    'borderRadius': '15px',
                    'background': 'linear-gradient(135deg, #9fa5d5, #e8f5c8)',  
                    'boxShadow': '0px 10px 20px rgba(240, 140, 50, 0.4)', 
                    'margin': '0 10px',
                }),
            ], style={
                'display': 'flex',
                'justifyContent': 'space-between',
                'marginTop': '5px',
                'marginBottom': '5px',
            }),

            html.Div([
                html.H3("Feasibility Checks", style={
                    'fontSize': '24px', 
                    'fontFamily': 'Arial, sans-serif',
                    'marginBottom': '20px'
                }),
                html.Div(id="feasibility-results", style={
                    'textAlign': 'left', 
                    'fontSize': '18px', 
                    'fontFamily': 'Arial, sans-serif',
                    'marginBottom': '20px',
                    'padding': '10px',
                    'backgroundColor': '#f4f4f9',
                    'borderRadius': '8px',
                    'boxShadow': '0px 6px 15px rgba(0,0,0,0.1)'
                })
            ], style={'marginTop': '20px'})


        ]),

        dcc.Tab(label='Summary', children=[
            html.Div([
                html.H3("Task Scheduling Summary", style={
                    'fontSize': '24px', 
                    'fontFamily': 'Arial, sans-serif',
                    'marginBottom': '20px'
                }),
                html.P("result dataframe download"),
                html.Div(id='summary-table'),  
            
                dcc.Store(id="result-store", storage_type="memory"),


                html.Button(
                    id="download-solution-button",
                    children="Download Solution as Excel File",
                    style={
                        'width': '100%',
                        'height': '50px',
                        'lineHeight': '50px',
                        'borderWidth': '1px',
                        'borderStyle': 'solid',
                        'borderRadius': '10px',
                        'backgroundColor': '#28a745',
                        'color': 'white',
                        'fontSize': '18px',
                        'cursor': 'pointer',
                        'marginTop': '20px',
                        'transition': 'background-color 0.3s',
                    }
                ),
                dcc.Download(id="download-solution-file") 
            ], style={'padding': '20px'})
        ])
    ])
])

#callback for download template
@app.callback(
    Output('download-template-file', 'data'),
    Input('download-template-button', 'n_clicks'),
    prevent_initial_call=True  # Prevent the callback doesn't fire before a button click
)
def handle_download_template(n_clicks):
    if n_clicks:
        output = download_template() 
        return dcc.send_bytes(output.getvalue(), "template.xlsx")
    return dash.no_update

# Download solution button
@app.callback(
    Output('download-solution-file', 'data'),
    Input('download-solution-button', 'n_clicks'),
    State('result-store', 'data'),  
    prevent_initial_call=True
)
def download_solution(n_clicks, result_data):
    if n_clicks and result_data:
        #Convert the stored JSON back into a DataFrame
        result_df = pd.DataFrame(result_data)
        output = io.BytesIO()
        result_df.to_excel(output, index=False, engine='openpyxl')
        output.seek(0)
        return dcc.send_bytes(output.getvalue(), "solution_machine_scheduling.xlsx")
    return dash.no_update


#Run model button callback function
@app.callback(
    [Output('output-status', 'children'),
     Output('output-graph', 'children'),
     Output('summary-table', 'children'),
     Output('tasks-late', 'children'),
     Output('tasks-on-time', 'children'),
     Output('max-tardiness', 'children'),
     Output('result-store', 'data'),  
     Output('input-data-preview', 'style')],
    [Input('run-model-button', 'n_clicks')],
    [State('upload-data', 'contents'),
     State('time-limit', 'value'),
     State('algorithm-selection', 'value'),
     State('precedence-check', 'value')]
)
def run_model_on_click(n_clicks, file_contents, time_limit, algorithm, precedence):
    if n_clicks == 0:
        return '', '', '', '', '', '', None, {'display': 'block'}

    if file_contents is None:
        return 'No file uploaded. Please upload an Excel file.', '', '', '', '', '', None, {'display': 'block'}

    content_type, content_string = file_contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        # Load task data
        tasks = load_task_data(io.BytesIO(decoded))

        # Start optimization
        start_time = time.time()
        if algorithm == "gurobi":
            model_results = optimize_with_gurobi(tasks, time_limit, precedence)
        elif algorithm == "pulp":
            model_results = optimize_with_pulp(tasks, time_limit, precedence)
        execution_time = time.time() - start_time

  
        # Compute metrics
        task_id = model_results['tasks']
        tasks_late = sum(1 for task in task_id if task['tardiness'] > 0)
        tasks_on_time = sum(1 for task in task_id if task['tardiness'] == 0)
        max_tardiness = max(task['tardiness'] for task in task_id)

        # Generate Gantt chart
        gantt_chart_div = generate_gantt_chart(model_results, tasks, execution_time)

        # Format results for the table
        result_df = format_results_to_df(model_results, tasks)
        table = html.Div([
            dcc.Graph(
                id='task-schedule-table',
                figure={
                    'data': [{
                        'type': 'table',
                        'header': {'values': result_df.columns.tolist()},
                        'cells': {'values': [result_df[col].tolist() for col in result_df.columns]}
                    }]
                }
            )
        ])
        return '', gantt_chart_div, table, tasks_late, tasks_on_time, max_tardiness, model_results, {'display': 'none'}
    except Exception as e:
        return f"Error processing the file: {str(e)}", '', '', '', '', '', None, {'display': 'block'}

        
@app.callback(
    [Output('file-status', 'children'),
     Output('input-data-preview', 'children')],
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename')]
)

def update_file_status_and_preview(file_contents, filename):
    if file_contents is None:
        return '', ''

    #upload file prevbiew table
    content_type, content_string = file_contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_excel(io.BytesIO(decoded), engine='openpyxl')  
        preview_table = dcc.Graph(
            id='input-data-table',
            figure={
                'data': [{
                    'type': 'table',
                    'header': {'values': df.columns.tolist()},
                    'cells': {'values': [df[col].tolist() for col in df.columns]}
                }]
            }
        )
        return f"File uploaded successfully: {filename}", preview_table
    except Exception as e:
        return f"Error processing the file: {str(e)}", ''

 
@app.callback(
    Output('feasibility-results', 'children'),
    [Input('run-model-button', 'n_clicks'),
     Input('result-store', 'data')],  # result-store is now an input
    State('upload-data', 'contents')
)
def update_feasibility_results(n_clicks, model_results, file_contents):
    if n_clicks == 0 or model_results is None or file_contents is None:
        return "No results to display. Please upload data and run the model."

    try:
        # Load task data
        content_type, content_string = file_contents.split(',')
        decoded = base64.b64decode(content_string)
        tasks = load_task_data(io.BytesIO(decoded))

        #check feasibility
        feasibility = check_feasibility(tasks, model_results)

        result_list = []
        for check, status in feasibility.items():
            icon = "✅" if status else "❌"
            result_list.append(html.Div(f"{icon} {check}"))

        return result_list
    except Exception as e:
        return f"Error running feasibility checks: {str(e)}"


def upload_and_process_file(contents, filename, time_limit, precedence, algorithm):
    if contents is None:
        return None, html.Div("No file uploaded.")

    try:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        uploaded_file = io.BytesIO(decoded)
        
        
        tasks = load_task_data(uploaded_file)
        
        start_time = time.time()
        if algorithm == "gurobi":
            results = optimize_with_gurobi(tasks, time_limit, precedence)
        elif algorithm == "pulp":
            results = optimize_with_pulp(tasks, time_limit, precedence)
        execution_time = time.time() - start_time

        gantt_chart_div = generate_gantt_chart(results, tasks, execution_time)

        return gantt_chart_div, results  
    except Exception as e:
        return None, html.Div(f"An error occurred: {str(e)}")

def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")

# Run the app
if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run_server(debug=True)