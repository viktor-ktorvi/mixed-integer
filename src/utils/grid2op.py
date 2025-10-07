from grid2op.Observation import BaseObservation

ACTION_DICT_TYPE = dict[str, dict[str, list[tuple[int, int]]]]


def get_empty_action_dict() -> ACTION_DICT_TYPE:
    return {"set_bus": {"lines_or_id": [], "lines_ex_id": [], "generators_id": [], "loads_id": []}}


def set_line_buses(
    line_ids: list[int], sub_ids: list[int], bus_ids: list[int], action_dict: ACTION_DICT_TYPE, obs: BaseObservation
) -> ACTION_DICT_TYPE:
    assert len(line_ids) == len(sub_ids) == len(bus_ids)
    for i in range(len(line_ids)):
        line_id = line_ids[i]
        sub_id = sub_ids[i]
        bus_id = bus_ids[i]
        if obs.line_or_to_subid[line_id] == sub_id:
            action_dict["set_bus"]["lines_or_id"].append((line_id, bus_id))
        elif obs.line_ex_to_subid[line_id] == sub_id:
            action_dict["set_bus"]["lines_ex_id"].append((line_id, bus_id))
        else:
            raise ValueError(f"{line_id=} doesn't connect to {sub_id=}.")

    return action_dict


def set_gen_buses(gen_ids: list[int], bus_ids: list[int], action_dict: ACTION_DICT_TYPE) -> ACTION_DICT_TYPE:
    assert len(gen_ids) == len(bus_ids)
    action_dict["set_bus"]["generators_id"] += list(zip(gen_ids, bus_ids))

    return action_dict


def set_load_buses(load_ids: list[int], bus_ids: list[int], action_dict: ACTION_DICT_TYPE) -> ACTION_DICT_TYPE:
    assert len(load_ids) == len(bus_ids)
    action_dict["set_bus"]["loads_id"] += list(zip(load_ids, bus_ids))

    return action_dict
