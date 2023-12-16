
model_config = dict(
    num_enti_cats   = 36,
    num_pred_cats   = 133,
    dim_ffn         = 512,
    dim_enti        = 512,
    dim_pred        = 512,
    dim_att         = 512,
    dim_feat        = 2048,       
    dim_clsme       = 300,
    enco_pool_len   = 4,
    n_enco_layers   = 2,
    n_deco_layers   = 6,
    n_att_head      = 8,
    num_querys      = 192,
    neg_weight      = 0.1,
    positive_vIoU_th= 0.5,
    EntiNameEmb_path= "vidvrd_EntiNameEmb_pku.npy",
    PredNameEmb_path = "vidvrd_PredNameEmb.npy",
    bias_matrix_path = "pred_bias_matrix_vidvrd_pku.npy",
    relation_path = "vidvrd_relation.pkl",
    concept_graph_path = "vidvrd_graph.pkl",
    cost_coeff_dict = dict(
        classification      = 1.0,
        adj_matrix          = 30.0,
    ),
    loss_coeff_dict = dict(         
        classification      = 1.0,
        adj_matrix          = 30.0,
    )
)

test_dataset_config = dict(
    split = "test",
    ann_dir = "datasets/vidvrd-dataset",
    proposal_dir = "datasets/preprocess_data/tracking/videovrd_detect_tracking",
    dim_boxfeature = 2048,
    min_frames_th = 5,
    max_proposal = 50,
    max_preds = 100,
    cache_tag = "PKU_v1",
    cache_dir = "datasets/cache"
)

inference_config = dict(
    topk = 10,
)
