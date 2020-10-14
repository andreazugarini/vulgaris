

def pld(varieties_ppl_dict):
    groups = list(varieties_ppl_dict.keys())
    _pld = {f"({g1},{g2})":0.0 for g1 in groups for g2 in groups}
    for g1 in groups:
        for g2 in groups:
            _pld[f"({g1},{g2})"] += varieties_ppl_dict[g1][g2]
            _pld[f"({g2},{g1})"] += varieties_ppl_dict[g1][g2]

    _pld = {k: v/2 for k,v in _pld.items()}

    return _pld
