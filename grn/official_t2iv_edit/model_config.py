import os.path as osp


def default_block_chunks_for_model(model_name: str) -> int:
    name = str(model_name)
    if 'GRN2b' in name:
        return 7
    raise ValueError(f'GRNEdit supports only the GRN 2B model, got {model_name!r}.')


def official_t2v_weight_candidates(weights_dir: str, model_name: str):
    weights_dir = osp.abspath(weights_dir)
    default_block_chunks_for_model(model_name)
    return [
        osp.join(weights_dir, 'GRN_T2V_2B.pth'),
        osp.join(weights_dir, 'GRN_T2V_2B_FSA_sft3800_non_ema.pth'),
    ]


def find_official_t2v_weight(weights_dir: str, model_name: str) -> str:
    for candidate in official_t2v_weight_candidates(weights_dir, model_name):
        if osp.isfile(candidate):
            return candidate
    return ''
