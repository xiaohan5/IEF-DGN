import argparse
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation 
from recbole.utils import init_logger, get_trainer, init_seed, set_color
import argparse
from model import *
import os

def run_single_model(args):
    #cur_dir = os.getcwd()
    dir = "/Users/hanzhihao/TriGAN/"
    config = Config(
        model=MSINE,
        dataset=args.dataset, 
        config_file_list=[dir+'props/overall.yaml', dir+'props/SINE.yaml']
    )
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)
    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = MSINE(config, train_data.dataset, args.num).to(config['device'])
    logger.info(model)
    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])
    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='diginetica', help='diginetica, nowplaying, retailrocket, tmall, yoochoose')
    args, _ = parser.parse_known_args()
    if args.dataset == 'tmall':
        args.num = 135
    elif args.dataset == 'diginetica':
        args.num = 310
    elif args.dataset == 'nowplaying':
        args.num = 433
    elif args.dataset == 'retailrocket':
        args.num = 348
    run_single_model(args)