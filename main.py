from config import config, paths, setup
from src.scripts import experiments, experiments_nick
from src.controller import d2_mask
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Input the routine you want to run.")
    parser.add_argument("--mode", default="CASAC", choices=["CASPL","CAS","AL","CASAC","Vanilla", "AC", "trial"], help="Select the mode you want to run on. CNS for consistency regularisation only. AL for active learning only. \
                        CAS for  both. CASPL for both with best 40 pseudolabels automatically trained. Vanilla for original incremental learning.")
    parser.add_argument("--iter", default=10, help="The number of iterations after initialisation you would like to run.")
    parser.add_argument("--train_from_init", action=argparse.BooleanOptionalAction, help="Whether to train an initial model or to increase from pre-trained steps.")
    parser.add_argument("--corrections", action=argparse.BooleanOptionalAction, help="Whether to correct pseudolabels with autocorrect features.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

# Determine the mode of running
    if args.mode == "CASAC":
        experiments.exp_240212_craac(iter=args.iter, train_from_init=args.train_from_init, corrections=args.corrections)
    elif args.mode == "CAS":
        experiments_nick.exp_0528_cas(iter=args.iter)
    elif args.mode == "AL":
        experiments.exp_240218_al(iter=args.iter)
    elif args.mode == "Vanilla":
        experiments.exp_240212_vanilla(iter=args.iter)
    elif args.mode == "CASPL":
        experiments_nick.exp_0528_caspl(iter=args.iter)
    elif args.mode == "AC":
        for i in range(1,28):
            experiments.AutocorrRoutine(
                it=i, 
                output_folder='./output/240212_craac', 
                model_weights=f'./output/240212_craac/weights/model_{str(i).zfill(2)}/model_final.pth',
                config_file = config.get_cfg_for_cns(**setup.exp_0528_cns_prediction)
            )
        # experiments.AutocorrRoutine(
        #     it=0, 
        #     output_folder='./output/240212_craac', 
        #     model_weights=f'./output/240212_craac/weights/init_model/model_final.pth',
        #     config_file = config.get_cfg_for_cns(**setup.exp_0528_cns_prediction)
        # )
    elif args.mode == "trial":
        for i in range(0,30):
            experiments.inferEval(
                it=i, 
                output_folder='./output/240214_al',
                model_weights=f'./output/240214_al/weights/model_{str(i).zfill(2)}/model_final.pth', 
                config_file = config.get_cfg_for_al(**setup.exp_0528_al_prediction)
            )
        # experiments.trial(
        #     it=18, 
        #     output_folder='./output/240112_cns',
        #     model_weights=f'./output/240112_cns/weights/model_{str(18).zfill(2)}/model_final.pth', 
        #     config_file = config.get_cfg_for_cns(**setup.exp_0528_cns_prediction)
        # )

    else:
        print("please input the routine you would like to run")

    