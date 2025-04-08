task_assignment_output_rules =  """
Please generate the output in the following json format:
{
  "reason": "<specific reason explanation>",
  "robot_id_task_pairs": [
    {
      "robot_id": <robot_id>,
      "action_chain": [
        "[action] <object_name> (object_id)",
        "[action] <object_name> (object_id)"
      ]
    },
    {
      "robot_id": <robot_id>,
      "action_chain": [
        "[action] <object_name> (object_id)",
        "[action] <object_name> (object_id)"
      ]
    },
  ]
}
"""

action_output_requirement = """
Please generate the output in the following json format:
{
    "inference_process": "The inference process of the action.",
    "reason": "The reason for choosing the action.",
    "step":  "The chosen action."
}
"""

action_chain_output_requirement = """
Please generate the output in the following json format:
{
    "inference_process": "The inference process of the action chain.",
    "intention": "The intention or reason of the action chain.",
    "action_chain": [
        "[action] <object_name> (object_id)",
        "[action] <object_name> (object_id)"
    ]
}
"""

refine_result_output_requirement = """
Please generate the output in the following json format:
{
    "inference_process": "The inference process of the better action chain.",
    "action_chain": [
        "[action] <object_name> (object_id)",
        "[action] <object_name> (object_id)"
    ]
}
"""

"""
Please generate the output in the following json format:
{
    "reason": "The reason for choosing the method. ",
    "method": "The method of refining. ",
    "action_chain": [
        "[action] <object_name> (object_id)",
        "[action] <object_name> (object_id)"
    ]
}
"""

conflict_solution_output_requirement = """
Please generate the output in the following json format:
{
    "reason_agent0": "The reason for agent0 choosing the method. ",
    "method_agent0": "The method of agent0 refining. ",
    "action_chain0": [
        "[action] <object_name> (object_id)",
        "[action] <object_name> (object_id)"
    ],
    "reason_agent1" : "The reason for agent1 choosing the method. ",
    "method_agent1": "The method of agent1 refining. ",
    "action_chain1": [
        "[action] <object_name> (object_id)",
        "[action] <object_name> (object_id)"
    ],
}
"""

action_output_requirement = """
Please generate the output in the following json format:
{
    "inference_process": "The inference process of the action.",
    "reason": "The reason for choosing the action.",
    "step":  "The chosen action."
    "action_chain": [
        "[action] <object_name> (object_id)",
        "[action] <object_name> (object_id)"
    ]
}
"""