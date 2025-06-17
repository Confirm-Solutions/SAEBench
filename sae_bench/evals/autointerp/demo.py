import os
import time
from dotenv import load_dotenv
from sae_lens import SAE
from transformer_lens import HookedTransformer

import sae_bench.sae_bench_utils.general_utils as general_utils
from sae_bench.evals.autointerp.eval_config import AutoInterpEvalConfig
from sae_bench.evals.autointerp.main import run_eval

if __name__ == "__main__":
    load_dotenv(override=True)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise Exception("Please set OPENAI_API_KEY in your .env file")

    device = general_utils.setup_environment()

    start_time = time.time()

    random_seed = 42
    output_folder = "eval_results/autointerp"

    model_name = "pythia-70m-deduped"
    hook_layer = 4

    sae = SAE.from_pretrained(
        "sae_bench_pythia70m_sweep_standard_ctx128_0712",
        "blocks.4.hook_resid_post__trainer_10",
    )
    selected_saes = [("sae_bench_pythia70m_sweep_standard_ctx128_0712", sae)]

    config = AutoInterpEvalConfig(
        random_seed=random_seed,
        model_name=model_name,
    )

    # create output folder
    os.makedirs(output_folder, exist_ok=True)

    # run the evaluation on all selected SAEs
    results_dict = run_eval(
        config,
        selected_saes,
        device,
        api_key,
        output_folder,
        force_rerun=True,
    )

    end_time = time.time()

    print(f"Finished evaluation in {end_time - start_time} seconds")
