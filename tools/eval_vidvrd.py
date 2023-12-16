

import root_path

import pickle
import json
import os
import argparse
from tqdm import tqdm
import torch

from dataloaders.dataloader_vidvrd import Dataset_pku
from models import UCML
from utils.evaluate import EvalFmtCvtor
from utils.utils_func import create_logger,parse_config_py
from VidVRDhelperEvalAPIs import eval_relation_with_gt
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
torch.set_printoptions(sci_mode=False,precision=4,linewidth=160)


def load_checkpoint(model,optimizer,scheduler,ckpt_path):
    checkpoint = torch.load(ckpt_path,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    scheduler.load_state_dict(checkpoint['sched_state_dict'])
    crt_epoch = checkpoint["crt_epoch"]
    batch_size = checkpoint["batch_size"]

    return model,optimizer,scheduler,crt_epoch,batch_size

def save_checkpoint(batch_size,crt_epoch,model,optimizer,scheduler,save_path):
    checkpoint = {
        "batch_size":batch_size,
        "crt_epoch":crt_epoch + 1,
        "model_state_dict":model.state_dict(),
        "optim_state_dict":optimizer.state_dict(),
        "sched_state_dict":scheduler.state_dict(),
    }
    torch.save(checkpoint,save_path)


def inference_then_eval(
    cfg_path,
    weight_path,
    gt_relations_path,
    save_tag="",    
    experiment_dir=None,
    gpu_id = 0,
    save_infer_result=False,
    save_relation_json=False
):

    device = torch.device("cuda:{}".format(gpu_id))

    if experiment_dir == None:
        experiment_dir = os.path.dirname(cfg_path)
    log_dir = os.path.join(experiment_dir,'logfile/')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    all_cfgs = parse_config_py(cfg_path)
    dataset_config = all_cfgs["test_dataset_config"]
    model_config = all_cfgs["model_config"]
    infer_config = all_cfgs["inference_config"]
    topk=infer_config["topk"]

    log_path = os.path.join(log_dir,'eval_{}.log'.format(save_tag))
    logger = create_logger(log_path)

    logger.info(model_config)
    logger.info(dataset_config)
    logger.info(infer_config)

    model = UCML(model_config,is_train=False)

    model = model.cuda(device)
    state_dict = torch.load(weight_path, map_location=torch.device('cpu'))

    state_dict_ = {}
    for k in state_dict.keys():
        if 'module' in k:
            state_dict_[k[7:]] = state_dict[k]
        else:
            state_dict_[k] = state_dict[k]

    model.load_state_dict(state_dict_, strict=False)
    model.eval()

    dataset = Dataset_pku(**dataset_config)        
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        drop_last=False,
        collate_fn=dataset.collator_func,
        shuffle=False,
        num_workers=2
    )

    logger.info("start inference...")
    logger.info("infer_config:{}".format(infer_config))
    logger.info("weight_path:{}".format(weight_path))

    convertor = EvalFmtCvtor("vidvrd")
    predict_relations = {}
    infer_result_for_save = {}
    for proposal_list,gt_graph_list in tqdm(dataloader):
        proposal_list = [p.to(device) for p in proposal_list]
        gt_graph_list = [g.to(device) for g in gt_graph_list]
        with torch.no_grad():
            batch_triplets = model(proposal_list,topk=topk)

        assert len(proposal_list) == 1
        proposal = proposal_list[0].to(torch.device("cpu"))
        video_name = proposal_list[0].video_name
        if batch_triplets[0] is None:
            infer_result = None
            continue

        (
            uniq_quintuples,  
            uniq_scores,      
            uniq_dura_inters, 
            uniq_query_ids,   
        ) = batch_triplets[0]
        uniq_scores = torch.mean(uniq_scores,dim=-1)  
        infer_result = (uniq_quintuples.cpu(),uniq_scores.cpu(),uniq_dura_inters.cpu())
        infer_result_for_save[video_name] = [x.cpu() for x in batch_triplets[0]]  

        pr_result = convertor.to_eval_format_pr(proposal,infer_result)
        predict_relations.update(pr_result)

    if save_infer_result:
        save_path = os.path.join(experiment_dir,'VidVRDtest_infer_result_{}.pkl'.format(save_tag))
        with open(save_path,'wb') as f:
            pickle.dump(infer_result_for_save,f)
        logger.info("infer_result saved at {}".format(save_path))

    eval_relation_with_gt(
        gt_relations_path=gt_relations_path,
        dataset_type="vidvrd",
        logger=logger,
        prediction_results=predict_relations
    )
    
    if save_relation_json:
        save_path = os.path.join(experiment_dir, 'VidVRDtest_predict_relations_{}.json'.format(save_tag))
        logger.info("saving predict_relations into {}...".format(save_path))
        with open(save_path,'w') as f:
            json.dump(predict_relations,f)
        logger.info("predict_relations have been saved at {}".format(save_path))
    logger.info("log file have been saved at {}".format(log_path))
    logger.handlers.clear()





if __name__ == "__main__":
    argvs = [
        'python', \
        '--cfg_path', 'experiments/config_.py', \
        '--ckpt_path', 'vidvrd_best.pth', \
        '--gt_relations_path', 'datasets/GT_json_for_eval/VidVRDtest_gts.json', \
        '--save_tag', 'vidvrd'
    ]
    sys.argv = argvs
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--cfg_path", type=str,help="...")
    parser.add_argument("--ckpt_path", type=str,help="...")
    parser.add_argument("--output_dir", type=str, help="...")
    parser.add_argument("--gt_relations_path", type=str, help="...")
    parser.add_argument("--cuda", type=int, default=0, help="...")
    parser.add_argument("--save_json_results", action="store_true",default=False,help="...")
    parser.add_argument("--save_tag", type=str,default="",help="...")
    parser.add_argument("--json_results_path", type=str,help="...")
    args = parser.parse_args()


    inference_then_eval(
        args.cfg_path,
        args.ckpt_path,
        gt_relations_path=args.gt_relations_path,
        save_tag=args.save_tag,
        experiment_dir=args.output_dir,
        gpu_id = args.cuda,
        save_infer_result=False,
        save_relation_json=False
        )