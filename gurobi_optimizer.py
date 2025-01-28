from gurobipy import Model, GRB, quicksum

def optimize_with_gurobi(tasks, time_limit, precedence):
    model = Model("Minimize_Weighted_Tardiness")

    #time limit
    model.Params.TimeLimit = time_limit

    #Decision variables
    start_times = {}
    tardiness = {}
    finish_times = {}
    no_overlap = {}
    task_self_overlap = {}

    #Initialize variables
    for _, task in tasks.iterrows():
        task_id = task['id']
        tardiness[task_id] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"T_{task_id}")
        finish_times[task_id] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"Finish_Time_{task_id}")

        for machine_idx in range(task['num_machines']):
            start_times[(task_id, machine_idx)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"S_{task_id}_{machine_idx}")

    #no-overlap variables for different tasks on the same machine
    for machine_idx in range(max(task['num_machines'] for _, task in tasks.iterrows())):
        for i, task_i in tasks.iterrows():
            for j, task_j in tasks.iterrows():
                if task_i['id'] != task_j['id']:
                    no_overlap[(task_i['id'], task_j['id'], machine_idx)] = model.addVar(
                        vtype=GRB.BINARY, name=f"y_{task_i['id']}_{task_j['id']}_{machine_idx}"
                    )

    #Initialize self-overlap variables for tasks on different machines
    if not precedence:
        for _, task in tasks.iterrows():
            task_id = task['id']
            for machine_idx in range(task['num_machines']):
                for other_machine_idx in range(machine_idx + 1, task['num_machines']):
                    task_self_overlap[(task_id, machine_idx, other_machine_idx)] = model.addVar(
                        vtype=GRB.BINARY, name=f"z_{task_id}_{machine_idx}_{other_machine_idx}"
                    )

    #Big M 
    big_M = sum(
        sum(task['service_times']) for _, task in tasks.iterrows()
    )

    #Objective
    model.setObjective(
        quicksum(task['weight'] * tardiness[task['id']] for _, task in tasks.iterrows()),
        GRB.MINIMIZE
    )

    #constraints
    for _, task in tasks.iterrows():
        task_id = task['id']
        release_date = task['release date']
        due_date = task['due date']
        service_times = task['service_times']

        #all tasks have to go through all machines
        for machine_idx in range(len(service_times)):
            model.addConstr(
                start_times[(task_id, machine_idx)] >= 0,
                name=f"Task_{task_id}_Processed_On_Machine_{machine_idx}"
            )

            #release date must be less than all start dates on all machines
            model.addConstr(
                start_times[(task_id, machine_idx)] >= release_date,
                name=f"Release_Date_Task_{task_id}_Machine_{machine_idx}"
            )

        #finish time is the maximum completion time for all machines
        for machine_idx in range(len(service_times)):
            model.addConstr(
                finish_times[task_id] >= start_times[(task_id, machine_idx)] + service_times[machine_idx],
                name=f"Finish_Time_{task_id}_Machine_{machine_idx}"
            )

        #Positive Tardiness constraint: T_i >= Finish_Time - Due_Date
        model.addConstr(
            tardiness[task_id] >= finish_times[task_id] - due_date,
            name=f"Tardiness_Task_{task_id}"
        )

        if not precedence:
            #no self-overlap
            for machine_idx in range(len(service_times)):
                for other_machine_idx in range(machine_idx + 1, len(service_times)):
                    model.addConstr(
                        start_times[(task_id, machine_idx)] + service_times[machine_idx] <=
                        start_times[(task_id, other_machine_idx)] + big_M * (1 - task_self_overlap[(task_id, machine_idx, other_machine_idx)]),
                        name=f"NoSelfOverlap_{task_id}_{machine_idx}_{other_machine_idx}_1"
                    )
                    model.addConstr(
                        start_times[(task_id, other_machine_idx)] + service_times[other_machine_idx] <=
                        start_times[(task_id, machine_idx)] + big_M * task_self_overlap[(task_id, machine_idx, other_machine_idx)],
                        name=f"NoSelfOverlap_{task_id}_{other_machine_idx}_{machine_idx}_2"
                    )
        else:
            #precedence constraint
            for machine_idx in range(1, len(service_times)):
                model.addConstr(
                    start_times[(task_id, machine_idx)] >= start_times[(task_id, machine_idx - 1)] + service_times[machine_idx - 1],
                    name=f"Precedence_{task_id}_Machine_{machine_idx}"
                )

    #no overlap on the same machine
    for machine_idx in range(max(task['num_machines'] for _, task in tasks.iterrows())):
        for i, task_i in tasks.iterrows():
            for j, task_j in tasks.iterrows():
                if task_i['id'] != task_j['id']:
                    model.addConstr(
                        start_times[(task_i['id'], machine_idx)] + task_i['service_times'][machine_idx] <=
                        start_times[(task_j['id'], machine_idx)] + big_M * (1 - no_overlap[(task_i['id'], task_j['id'], machine_idx)]),
                        name=f"NoOverlap_{task_i['id']}_{task_j['id']}_Machine_{machine_idx}_1"
                    )
                    model.addConstr(
                        start_times[(task_j['id'], machine_idx)] + task_j['service_times'][machine_idx] <=
                        start_times[(task_i['id'], machine_idx)] + big_M * no_overlap[(task_i['id'], task_j['id'], machine_idx)],
                        name=f"NoOverlap_{task_j['id']}_{task_i['id']}_Machine_{machine_idx}_2"
                    )

    #sovle
    model.optimize()

    results = {
        "status": model.Status,
        "tasks": []
    }
    if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
        for _, task in tasks.iterrows():
            task_id = task['id']
            results['tasks'].append({
                "id": task_id,
                "tardiness": tardiness[task_id].X,
                "start_times": [start_times[(task_id, machine_idx)].X for machine_idx in range(task['num_machines'])],
                "finish_time": finish_times[task_id].X
            })

    return results