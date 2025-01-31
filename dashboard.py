import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import time
import webbrowser
from threading import Timer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import io
import base64
from gurobi_optimizer import optimize_with_gurobi
from pulp_optimizer import optimize_with_pulp


def format_number(value):
    """
    Format a number to show integers as is and decimals with 1 decimal place.
    """
    if pd.isna(value):
        return ''
    
    num = float(value)
    
    if num.is_integer():
        return str(int(num))
    else:
        return f"{num:.1f}"

def format_dataframe_values(df):
    """
    Format all numeric values in a DataFrame according to the format_number rules.
    """
    formatted_df = df.copy()
    
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            formatted_df[column] = df[column].apply(format_number)
    
    return formatted_df

def preview_table_with_formatting(df):
    """
    Create a formatted preview table using dcc.Graph.
    """
    formatted_df = format_dataframe_values(df)
    
    return dcc.Graph(
        id='input-data-table',
        figure={
            'data': [{
                'type': 'table',
                'header': {'values': formatted_df.columns.tolist()},
                'cells': {'values': [formatted_df[col].tolist() for col in formatted_df.columns]}
            }]
        }
    )

def summary_table_with_formatting(result_df):
    """
    Create a formatted summary table using dcc.Graph.
    """
    formatted_df = format_dataframe_values(result_df)
    
    return html.Div([
        dcc.Graph(
            id="task-schedule-table",
            figure={
                "data": [{
                    "type": "table",
                    "header": {"values": formatted_df.columns.tolist()},
                    "cells": {"values": [formatted_df[col].tolist() for col in formatted_df.columns]}
                }]
            }
        )
    ])


app = dash.Dash(__name__)   

#Function to download Excel file template
def download_template():
    output = io.BytesIO()
    df = pd.DataFrame(columns=["release date", "due date", "weight", "st on machine 1", "st on machine 2", "st on machine 3"])
    df.to_excel(output, index=False, engine='openpyxl')
    output.seek(0)
    return output

#Function to load task data from the uploaded file
def load_task_data(uploaded_file):
    df = pd.read_excel(uploaded_file, engine="openpyxl")
    
    tasks = df[['release date', 'due date', 'weight']].copy()
    tasks['id'] = range(1, len(tasks) + 1)

    service_time_columns = [col for col in df.columns if col.startswith('st')]
    tasks['service_times'] = df[service_time_columns].values.tolist()
    tasks['num_machines'] = len(service_time_columns)

    return tasks 

def generate_gantt_chart(results, tasks, execution_time):
    gantt_data = []
    total_weighted_tardiness = 0  
    
    task_names = list(set([f"Task {task['id']}" for task in results['tasks']]))
    
    cmap = plt.get_cmap("tab20")
    num_tasks = len(task_names)
    custom_colors = [matplotlib.colors.rgb2hex(cmap(i / num_tasks)) for i in range(num_tasks)]

    task_color_map = {task: custom_colors[i] for i, task in enumerate(task_names)}
            
    #Prepare the data 
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
    
    fig = go.Figure()
    fig_data = []

    #Add bars for each task and machine
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

        task_data = tasks[tasks['id'] == task_id].iloc[0]
        release_date = task_data['release date']
        due_date = task_data['due date']
        weight = task_data['weight']
        service_times = task_data['service_times']

        #Calculate the Done Date
        done_date = max(start_times[i] + service_times[i] for i in range(len(service_times)))

        #Calculate Tardiness = max(0, (Done Date - Due Date) * Weight)
        tardiness = max(0, (done_date - due_date) * weight)

       
        task_row = {
            'Task ID': task_id,
            'Release Date': round(release_date, 1),
            'Due Date': round(due_date, 1),
            'Weight': round(weight, 1),
        }

        #Add Start and Finish Times for each machine
        for machine_idx, (start_time, service_time) in enumerate(zip(start_times, service_times)):
            task_row[f"Start Time M{machine_idx + 1}"] = round(start_time, 1)
            task_row[f"Finish Time M{machine_idx + 1}"] = round(start_time + service_time, 1)
            #task_row[f"Service Time M{machine_idx + 1}"] = round(service_time)  # Optional for clarity

        #Add Done Date and Tardiness at the end
        task_row['Done Date'] = round(done_date, 1)
        task_row['Tardiness'] = round(tardiness, 1)

        task_details.append(task_row)

    df = pd.DataFrame(task_details)

    columns = [col for col in df.columns if col not in ['Done Date', 'Tardiness']]
    columns += ['Done Date', 'Tardiness'] 
    df = df[columns]

    return df

def check_feasibility(tasks, schedule_results):

    checks = {
        "Release dates respected": True,
        "No overlapping tasks on machines": True,
        "Due dates and tardiness correctly calculated": True,
    }

    #Check if release dates are respected
    for task in schedule_results["tasks"]:
        task_data = tasks[tasks["id"] == task["id"]].iloc[0]
        if task["start_times"][0] < task_data["release date"]:
            checks["Release dates respected"] = False
            break

    #Check for no overlapping tasks on machines
    machine_schedules = {}
    for task in schedule_results["tasks"]:
        task_data = tasks[tasks["id"] == task["id"]].iloc[0]  
        for idx, start_time in enumerate(task["start_times"]):
            service_time = task_data["service_times"][idx]
            finish_time = start_time + service_time
            machine_schedules.setdefault(idx, []).append((start_time, finish_time))


    for machine, intervals in machine_schedules.items():
        intervals.sort()
        for i in range(len(intervals) - 1):
            if intervals[i][1] > intervals[i + 1][0]:
                checks["No overlapping tasks on machines"] = False
                break

    #Check due dates and tardiness calculations
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
    # Tabs for navigation
    dcc.Tabs([
        # First Tab: Task Scheduling
        dcc.Tab(label='Task Scheduling', children=[
            html.Div([
                # Left block for input data & constraints
                html.Div([
                    html.H3("Input Data & Constraints", style={
                        'fontSize': '24px', 
                        'fontFamily': 'Arial, sans-serif', 
                        'marginBottom': '10px'
                    }),
                    # Add Download Template Button
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
                    dcc.Download(id="download-template-file"),  # Add dcc.Download component for file download
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
                    # Add error message div
                    html.Div(id="validation-error", style={
                        'color': '#dc3545',
                        'marginBottom': '20px',
                        'fontSize': '16px',
                        'fontFamily': 'Arial, sans-serif'
                    }),
                    # Single file status div (removed duplicate)
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
                            {"label": "Gurobi Algorithm (license needed for local use)", "value": "gurobi"},
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
                    # Run Model button
                    html.Button(
                        id='run-model-button',
                        n_clicks=0,
                        children='Optimize schedule',
                        disabled=True,  # Initially disabled
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
                            'boxShadow': '0px 4px 10px rgba(0,0,0,0.1)',
                            'opacity': '0.5'  # Initially dimmed
                        }
                    ),
                    # Add a store component to track validation state
                    dcc.Store(id='validation-store', data={'is_valid': False}),
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

                # Right block for the plot
                html.Div([
                    html.H3(id="schedule-title", children="Input Data", style={
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
            ], style={'display': 'flex', 'justifyContent': 'space-between'}),

            html.Div(
                id="timer-container",
                children="Elapsed Time: 0 seconds",
                style={
                    "position": "absolute",
                    "top": "10px",
                    "right": "10px",
                    "fontSize": "16px",
                    "fontWeight": "bold",
                    "backgroundColor": "#f9f9f9",
                    "padding": "8px",
                    "borderRadius": "5px",
                    "boxShadow": "0px 2px 5px rgba(0,0,0,0.1)",
                    "zIndex": "1000"
                }
            ),
            dcc.Store(id="elapsed-time", data=0),
            dcc.Interval(
                id="timer-interval",
                interval=1000,
                n_intervals=0,
                disabled=True
            ),

            # Three blocks under the data input and constraints section
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
                    'background': 'rgba(173, 216, 230, 0.7)',
                    'boxShadow': '0px 10px 20px rgba(0, 0, 0, 0.1)',
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
                    'background': 'rgba(173, 216, 230, 0.7)',
                    'boxShadow': '0px 10px 20px rgba(0, 0, 0, 0.1)',
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
                    'background': 'rgba(173, 216, 230, 0.7)',
                    'boxShadow': '0px 10px 20px rgba(0, 0, 0, 0.1)',
                    'margin': '0 10px',
                }),
            ], style={
                'display': 'flex',
                'justifyContent': 'space-between',
                'marginTop': '20px',
                'marginBottom': '20px',
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
                
                html.Div(id='summary-table'),  # The table will be rendered here
            
                dcc.Store(id="result-store", storage_type="memory"),

                # Add Download Solution Button
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

def validate_file_format(contents, filename):
    if contents is None:
        return False, "No file uploaded."
    
    if not filename.endswith('.xlsx'):
        return False, "Invalid file format. Please upload an Excel (.xlsx) file."
    
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_excel(io.BytesIO(decoded), engine='openpyxl')
        
        #First check if there are any data rows at all
        if len(df) == 0:
            return False, "File contains no data rows. At least one task is required."
        
        #Check for required columns
        required_columns = ["release date", "due date", "weight"]
        service_time_columns = [col for col in df.columns if col.startswith('st')]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
        
        if not service_time_columns:
            return False, "No service time columns found. Column names should start with 'st'"
        
        #Check if there's at least one row with valid numerical data
        numeric_columns = required_columns + service_time_columns
        try:
            #Convert to numeric, coercing errors to NaN
            numeric_df = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
            
            #Check if there's at least one complete row (all values are numeric)
            valid_rows = numeric_df.notna().all(axis=1)
            num_valid_rows = valid_rows.sum()
            
            if num_valid_rows == 0:
                return False, "No valid tasks found. Each task must have complete numerical values for all required fields."
            
            #Now convert to numeric, which we know will work for at least one row
            df[numeric_columns] = numeric_df
            
            #Check for negative values
            if (df[numeric_columns] < 0).any().any():
                return False, "Negative values found in numeric columns"
            
            return True, f"File format is valid. Found {num_valid_rows} valid task(s)."
            
        except ValueError as e:
            return False, f"Error processing numeric values: {str(e)}"
            
    except Exception as e:
        return False, f"Error processing file: {str(e)}"


@app.callback(
    [Output("validation-error", "children", allow_duplicate=True),
     Output("file-status", "children", allow_duplicate=True),
     Output("input-data-preview", "children", allow_duplicate=True),
     Output("run-model-button", "disabled", allow_duplicate=True),
     Output("run-model-button", "style", allow_duplicate=True),
     Output("validation-store", "data")],
    [Input("upload-data", "contents"),
     Input("upload-data", "filename")],
    prevent_initial_call=True
)
def update_validation(contents, filename):
    if contents is None:
        return "", "", None, True, {
            'width': '100%',
            'height': '50px',
            'lineHeight': '50px',
            'borderRadius': '10px',
            'background': 'blue',
            'color': 'white',
            'fontSize': '20px',
            'cursor': 'not-allowed',
            'transition': 'background-color 0.3s',
            'border': 'none',
            'marginTop': '20px',
            'boxShadow': '0px 4px 10px rgba(0,0,0,0.1)',
            'opacity': '0.5'
        }, {'is_valid': False}

    is_valid, message = validate_file_format(contents, filename)
    
    if is_valid:
        #Create preview table for valid file
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
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
        
        button_style = {
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
            'boxShadow': '0px 4px 10px rgba(0,0,0,0.1)',
            'opacity': '1'
        }
        
        return "", f"File uploaded successfully: {filename}", preview_table, False, button_style, {'is_valid': True}
    else:
        button_style = {
            'width': '100%',
            'height': '50px',
            'lineHeight': '50px',
            'borderRadius': '10px',
            'background': 'blue',
            'color': 'white',
            'fontSize': '20px',
            'cursor': 'not-allowed',
            'transition': 'background-color 0.3s',
            'border': 'none',
            'marginTop': '20px',
            'boxShadow': '0px 4px 10px rgba(0,0,0,0.1)',
            'opacity': '0.5'
        }
        
        return message, "", None, True, button_style, {'is_valid': False}

@app.callback(
    [Output("timer-interval", "disabled", allow_duplicate=True),
     Output("timer-container", "children", allow_duplicate=True),
     Output("elapsed-time", "data", allow_duplicate=True)],
    [Input("run-model-button", "n_clicks"),
     Input("timer-interval", "n_intervals")],
    [State("timer-interval", "disabled"),
     State("elapsed-time", "data")],
    prevent_initial_call=True
)
def manage_timer(n_clicks, n_intervals, timer_disabled, elapsed_time):
    if n_clicks > 0 and timer_disabled:
        return False, "Elapsed Time: 0 seconds", 0
    if not timer_disabled:
        return dash.no_update, f"Elapsed Time: {n_intervals} seconds", n_intervals
    return True, f"Elapsed Time: {elapsed_time} seconds", elapsed_time


@app.callback(
    Output("timer-container", "children"),
    Input("elapsed-time", "data")
)
def display_elapsed_time(elapsed_time):
    return f"Elapsed Time: {elapsed_time} seconds"



#Download template button
@app.callback(
    Output('download-template-file', 'data'),
    Input('download-template-button', 'n_clicks'),
    prevent_initial_call=True  #ensures the callback doesn't fire before a button click
)
def handle_download_template(n_clicks):
    if n_clicks:
        output = download_template() 
        return dcc.send_bytes(output.getvalue(), "template.xlsx")
    return dash.no_update


@app.callback(
    Output('download-solution-file', 'data'),
    Input('download-solution-button', 'n_clicks'),
    State('result-store', 'data'), 
    prevent_initial_call=True
)
def download_solution(n_clicks, result_data):
    if n_clicks and result_data:
        result_df = pd.DataFrame(result_data)
        output = io.BytesIO()
        result_df.to_excel(output, index=False, engine='openpyxl')
        output.seek(0)
        return dcc.send_bytes(output.getvalue(), "solution_machine_scheduling.xlsx")
    return dash.no_update


# Main callback 
@app.callback(
    [Output("output-graph", "children"),
     Output("input-data-preview", "style"),
     Output("input-data-preview", "children", allow_duplicate=True),
     Output("file-status", "children", allow_duplicate=True),
     Output("tasks-late", "children"),
     Output("tasks-on-time", "children"),
     Output("max-tardiness", "children"),
     Output("elapsed-time", "data", allow_duplicate=True),
     Output("timer-interval", "disabled", allow_duplicate=True),
     Output("output-status", "children"),
     Output("summary-table", "children"),
     Output("result-store", "data"),
     Output("feasibility-results", "children"),
     Output("schedule-title", "children"),
     Output("validation-error", "children", allow_duplicate=True),
     Output("run-model-button", "disabled", allow_duplicate=True),
     Output("run-model-button", "style", allow_duplicate=True)],
    [Input("upload-data", "contents"),
     Input("upload-data", "filename"),
     Input("run-model-button", "n_clicks")],
    [State("time-limit", "value"),
     State("algorithm-selection", "value"),
     State("precedence-check", "value"),
     State("validation-store", "data")], 
    prevent_initial_call=True
)
def handle_file_upload_and_run_model(contents, filename, n_clicks, time_limit, algorithm, precedence, validation_state):
    ctx = dash.callback_context
    triggered_input = ctx.triggered[0]["prop_id"].split(".")[0]  

    disabled_button_style = {
        'width': '100%',
        'height': '50px',
        'lineHeight': '50px',
        'borderRadius': '10px',
        'background': 'blue',
        'color': 'white',
        'fontSize': '20px',
        'cursor': 'not-allowed',
        'transition': 'background-color 0.3s',
        'border': 'none',
        'marginTop': '20px',
        'boxShadow': '0px 4px 10px rgba(0,0,0,0.1)',
        'opacity': '0.5'
    }

    enabled_button_style = dict(disabled_button_style)
    enabled_button_style.update({
        'cursor': 'pointer',
        'opacity': '1'
    })

    # if not validation_state['is_valid']:
    #     raise dash.exceptions.PreventUpdate

    # file upload
    if triggered_input == "upload-data":
        if contents is None:
            return (None, {"display": "block"}, "", "", "", "", "", 0, True, "", "", {}, "", 
                    "Input Data", "No file uploaded", True, disabled_button_style)

        #Validate file
        is_valid, validation_message = validate_file_format(contents, filename)
        
        if not is_valid:
            return (None, {"display": "block"}, "", "", "", "", "", 0, True, "", "", {}, "", 
                    "Input Data", validation_message, True, disabled_button_style)

        #If valid, show preview
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_excel(io.BytesIO(decoded), engine='openpyxl')
            preview_table = preview_table = preview_table_with_formatting(df)  
            return (None, {"display": "block"}, preview_table, 
                    f"File uploaded successfully: {filename}", "", "", "", 0, True, "", "", {}, "", 
                    "Input Data", "", False, enabled_button_style)

        except Exception as e:
            return (None, {"display": "block"}, "", f"Error processing the file: {str(e)}", 
                    "", "", "", 0, True, "", "", {}, "", "Input Data", str(e), True, disabled_button_style)

    # optimize button click
    elif triggered_input == "run-model-button":
        if contents is None:
            return (None, dash.no_update, dash.no_update, "No file uploaded", "", "", "", 0, True, "", "", {}, "", 
                    "Input Data", "Please upload a file first", True, disabled_button_style)

        try:
            
            tasks = load_task_data(io.BytesIO(base64.b64decode(contents.split(",")[1])))
            start_time = time.time()

            if algorithm == "gurobi":
                model_results = optimize_with_gurobi(tasks, time_limit, precedence)
            else:
                model_results = optimize_with_pulp(tasks, time_limit, precedence)

            execution_time = round(time.time() - start_time)

           
            gantt_chart_div = generate_gantt_chart(model_results, tasks, execution_time)
            result_df = format_results_to_df(model_results, tasks)

            table = summary_table_with_formatting(result_df) 

            task_id = model_results['tasks']
            num_late_tasks = sum(1 for task in task_id if task['tardiness'] > 0)
            num_on_time_tasks = sum(1 for task in task_id if task['tardiness'] == 0)
            task_id_to_weight = dict(zip(tasks['id'], tasks['weight']))
            max_tardiness = max(
                task['tardiness'] * task_id_to_weight[task['id']] 
                for task in task_id
            )

            feasibility_results = ""
            feasibility_passed = True

            #Check release dates
            release_dates_respected = all(
                task["start_times"][0] >= tasks.loc[tasks['id'] == task["id"], 'release date'].values[0]
                for task in model_results["tasks"]
            )

            #Check no overlapping tasks
            no_overlaps = True
            machine_schedules = {}
            for task in model_results["tasks"]:
                task_data = tasks[tasks['id'] == task["id"]].iloc[0]
                for idx, start_time in enumerate(task["start_times"]):
                    service_time = task_data["service_times"][idx]
                    finish_time = start_time + service_time
                    machine_schedules.setdefault(idx, []).append((start_time, finish_time))

            for machine, intervals in machine_schedules.items():
                intervals.sort()
                for i in range(len(intervals) - 1):
                    if intervals[i][1] > intervals[i + 1][0]:
                        no_overlaps = False
                        break
                if not no_overlaps:
                    break

            #Check tardiness calculations
            tardiness_correct = all(
                abs(task["tardiness"] - max(0, max(start + service for start, service in zip(task["start_times"], tasks.loc[tasks['id'] == task["id"], 'service_times'].values[0])) - tasks.loc[tasks['id'] == task["id"], 'due date'].values[0])) < 1e-5
                for task in model_results["tasks"]
            )

            feasibility_passed = all([release_dates_respected, no_overlaps, tardiness_correct])

            feasibility_results = dcc.Markdown(
                ""
                + ("❌ Release dates violated.\n\n" if not release_dates_respected else "✅ Release dates respected.\n\n")
                + ("❌ Overlapping tasks detected.\n\n" if not no_overlaps else "✅ No overlapping tasks on machines.\n\n")
                + ("❌ Incorrect tardiness calculations.\n\n" if not tardiness_correct else "✅ Due dates and tardiness correctly calculated.\n\n")
            )

            return (gantt_chart_div, {"display": "none"}, dash.no_update, "",
                    num_late_tasks, num_on_time_tasks, max_tardiness, execution_time,
                    True, "", table, result_df.to_dict("records"), feasibility_results,
                    "Feasible Solution ✅" if feasibility_passed else "Infeasible Solution ❌",
                    "", False, enabled_button_style)

        except Exception as e:
            return (None, dash.no_update, dash.no_update, f"Error: {str(e)}", "", "", "", 0, True,
                    "", "", {}, "", "Input Data", str(e), True, disabled_button_style)

    return dash.no_update



@app.callback(
        Output("timer-container", "children", allow_duplicate=True),
        Output("timer-interval", "n_intervals"),
        Input("run-model-button", "n_clicks"),
        prevent_initial_call = True
)
def reset_timer(n_clicks):
     if n_clicks > 0:
        return "Elapsed Time: 0 seconds", 0
     return dash.no_update, dash.no_update


def update_file_status_and_preview(file_contents, filename):
    if file_contents is None:
        return '', ''

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


def upload_and_process_file(contents, filename, time_limit, precedence, algorithm):
    if contents is None:
        return None, html.Div("No file uploaded.")

    try:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        uploaded_file = io.BytesIO(decoded)
        
        #Load tasks
        tasks = load_task_data(uploaded_file)
        
        #Solve the optimization problem
        start_time = time.time()
        if algorithm == "gurobi":
            results = optimize_with_gurobi(tasks, time_limit, precedence)
        elif algorithm == "pulp":
            results = optimize_with_pulp(tasks, time_limit, precedence)
        execution_time = time.time() - start_time

        #Generate Gantt chart using Plotly
        gantt_chart_div = generate_gantt_chart(results, tasks, execution_time)

        return gantt_chart_div, results  
    except Exception as e:
        return None, html.Div(f"An error occurred: {str(e)}")

#Run the Dash app
if __name__ == "__main__":
    Timer(1, lambda: webbrowser.open_new("http://127.0.0.1:8050/")).start()
    app.run_server(debug=True)  