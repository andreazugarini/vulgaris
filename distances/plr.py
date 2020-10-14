

def plr(varieties_ppl_dict):
    groups = list(varieties_ppl_dict.keys())

    _plr = {}
    for g1 in groups:
        for g2 in groups:
            _plr[f"({g1},{g2})"] = varieties_ppl_dict[g1][g2]/varieties_ppl_dict[g2][g1]

    return _plr