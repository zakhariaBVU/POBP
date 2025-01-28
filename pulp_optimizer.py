from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, PULP_CBC_CMD

def optimize_with_pulp(tasks, time_limit, precedence):
    prob = LpProblem("Minimize_Weighted_Tardiness", LpMinimize)

    #Decision variables
    start_times = {}
    tardiness = {}
    no_overlap = {}
    self_overlap = {}
    finish_times = {}

    #create variables for each task and starting time for each (task, machine) pair
    for _, task in tasks.iterrows():
        task_id = task['id']
        tardiness[task_id] = LpVariable(f"T_{task_id}", lowBound=0, cat="Continuous")
        for machine_idx in range(task['num_machines']):
            start_times[(task_id, machine_idx)] = LpVariable(f"S_{task_id}_{machine_idx}", lowBound=0, cat="Continuous")

        finish_times[task_id] = LpVariable(f"F_{task_id}", lowBound=0, cat="Continuous")

    #overlap variables for tasks on different machines (self-overlap)
    for _, task in tasks.iterrows():
        task_id = task['id']
        num_machines = task['num_machines']
        for machine_idx in range(num_machines):
            for other_machine_idx in range(machine_idx + 1, num_machines):
                self_overlap[(task_id, machine_idx, other_machine_idx)] = LpVariable(
                    f"z_{task_id}_{machine_idx}_{other_machine_idx}", cat="Binary"
                )

    #overlap variables for different tasks on the same machine
    for machine_idx in range(tasks['num_machines'].max()):
        for i, task_i in tasks.iterrows():
            for j, task_j in tasks.iterrows():
                if task_i['id'] != task_j['id']:
                    no_overlap[(task_i['id'], task_j['id'], machine_idx)] = LpVariable(
                        f"y_{task_i['id']}_{task_j['id']}_{machine_idx}", cat="Binary"
                    )

    # Big M
    big_M = sum(
        max(task['service_times']) for _, task in tasks.iterrows()
    ) * tasks['num_machines'].max()

    #objective
    prob += lpSum(task['weight'] * tardiness[task['id']] for _, task in tasks.iterrows())

    #constraints
    for _, task in tasks.iterrows():
        task_id = task['id']
        release_date = task['release date']
        due_date = task['due date']
        service_times = task['service_times']

        #release date is less than or equal to start time
        for machine_idx in range(len(service_times)):
            prob += start_times[(task_id, machine_idx)] >= release_date, f"Release_Date_Task_{task_id}_Machine_{machine_idx}"

        #positive tardiness constraint: T_i >= F_i - D_i
        prob += finish_times[task_id] == start_times[(task_id, len(service_times) - 1)] + service_times[-1], f"FinishTime_Task_{task_id}"

        #positive tardiness constraint: T_i >= F_i - D_i
        prob += tardiness[task_id] >= finish_times[task_id] - due_date, f"Tardiness_Task_{task_id}"

        if precedence:
            #precedence constraints
            for machine_idx in range(1, len(service_times)):
                prob += (
                    start_times[(task_id, machine_idx)] >= start_times[(task_id, machine_idx - 1)] + service_times[machine_idx - 1],
                    f"Precedence_Task_{task_id}_Machine_{machine_idx}"
                )

        #prevent self-overlap for the same task across different machines
        for machine_idx in range(len(service_times)):
            for other_machine_idx in range(machine_idx + 1, len(service_times)):
                prob += (
                    start_times[(task_id, machine_idx)] + service_times[machine_idx]
                    <= start_times[(task_id, other_machine_idx)]
                    + big_M * (1 - self_overlap[(task_id, machine_idx, other_machine_idx)]),
                    f"SelfOverlap_Task_{task_id}_{machine_idx}_{other_machine_idx}_1"
                )
                prob += (
                    start_times[(task_id, other_machine_idx)] + service_times[other_machine_idx]
                    <= start_times[(task_id, machine_idx)]
                    + big_M * self_overlap[(task_id, machine_idx, other_machine_idx)],
                    f"SelfOverlap_Task_{task_id}_{other_machine_idx}_{machine_idx}_2"
                )

    #no overlap constraints for different tasks on the same machine
    for machine_idx in range(tasks['num_machines'].max()):
        for i, task_i in tasks.iterrows():
            for j, task_j in tasks.iterrows():
                if task_i['id'] != task_j['id']:
                    task_i_id = task_i['id']
                    task_j_id = task_j['id']
                    service_time_i = task_i['service_times'][machine_idx]
                    service_time_j = task_j['service_times'][machine_idx]

                    #task i finishes before Task j starts OR Task j finishes before Task i starts
                    prob += (
                        start_times[(task_i_id, machine_idx)] + service_time_i
                        <= start_times[(task_j_id, machine_idx)]
                        + (big_M * (1 - no_overlap[(task_i_id, task_j_id, machine_idx)])),
                        f"NoOverlap_{task_i_id}_{task_j_id}_{machine_idx}_1"
                    )
                    prob += (
                        start_times[(task_j_id, machine_idx)] + service_time_j
                        <= start_times[(task_i_id, machine_idx)]
                        + (big_M * no_overlap[(task_i_id, task_j_id, machine_idx)]),
                        f"NoOverlap_{task_i_id}_{task_j_id}_{machine_idx}_2"
                    )

    #solve
    prob.solve(PULP_CBC_CMD(timeLimit=time_limit))

    #results
    results = {
        "status": LpStatus[prob.status],
        "tasks": []
    }
    for _, task in tasks.iterrows():
        task_id = task['id']
        results['tasks'].append({
            "id": task_id,
            "tardiness": tardiness[task_id].varValue,
            "finish_time": finish_times[task_id].varValue, 
            "start_times": [start_times[(task_id, machine_idx)].varValue for machine_idx in range(task['num_machines'])]
        })

    return results