dataset_paths = {
	"lpff_flick_path" :"/raid/xjd/dataset/lpff/flick/eg3d", 
	"lpff_unslash_path" : "/raid/xjd/dataset/lpff/unslash-pexels-dataset/eg3d",
    "ffhq512": "/raid/xjd/workspace/eg3d/dataset_preprocessing/ffhq/final_crops",
}

model_paths = {
	# 'encoder_render': 'pretrained_models/encoder_render.pt',
	'encoder_render': 'pretrained_models/encoder_render_normal_140000.pt',
	'eg3d_ffhq': 'pretrained_models/ffhq512-128.pth',
	'ir_se50': 'pretrained_models/model_ir_se50.pth',
	'eg3d_rebalanced': 'pretrained_models/ffhqrebalanced512-128.pth',
	'InceptionV3': 'pretrained_models/pt_inception-2015-12-05-6726825d.pth',
    'lpff_weighs': 'pretrained_models/var2-128.pth',
}

val_paths = {
	'ffhq_mean': 'assets/ffhq_sample_mean.npy',
    'ffhq_cov': 'assets/ffhq_sample_cov.npy',
	# 'ffhq_mean': 'assets/ffhq_mean.npy',
    # 'ffhq_cov': 'assets/ffhq_cov.npy',
}

pretrained_model_path = model_paths['eg3d_rebalanced']
