dataset_paths = {
    'CUHK_train_P': '/media/gpu/T7/HifiSketch/datasets/CUHK_train_Photo',
    'CUHK_train_S': '/media/gpu/T7/HifiSketch/datasets/CUHK_train_Sketch',
	'CUHK_test_P': '/media/gpu/T7/HifiSketch/datasets/CUHK_test_Photo',
    'CUHK_test_S': '/media/gpu/T7/HifiSketch/datasets/CUHK_test_Sketch',

	'CUFSF_train_P': '/media/gpu/T7/HifiSketch/datasets/CUFSF_train_Photo',
    'CUFSF_train_S': '/media/gpu/T7/HifiSketch/datasets/CUFSF_train_Sketch',
	'CUFSF_test_P': '/media/gpu/T7/HifiSketch/datasets/CUFSF_test_Photo',
    'CUFSF_test_S': '/media/gpu/T7/HifiSketch/datasets/CUFSF_test_Sketch',

	'FS2K_train_P': '/media/gpu/T7/HifiSketch/datasets/FS2K_train_Photo',
    'FS2K_train_S': '/media/gpu/T7/HifiSketch/datasets/FS2K_trian_Sketch',
	'FS2K_test_P': '/media/gpu/T7/HifiSketch/datasets/FS2K_test_Photo',
    'FS2K_test_S': '/media/gpu/T7/HifiSketch/datasets/FS2K_test_Sketch',

	'WildSketch_train_P': '/media/gpu/T7/HifiSketch/datasets/WildSketch_trian_Photo',
    'WildSketch_train_S': '/media/gpu/T7/HifiSketch/datasets/WildSketch_trian_Sketch',
	'WildSketch_test_P': '/media/gpu/T7/HifiSketch/datasets/WildSketch_test_Photo',
    'WildSketch_test_S': '/media/gpu/T7/HifiSketch/datasets/WildSketch_test_Sketch',

}

model_paths = {
	# models for backbones and losses
	'ir_se50': 'pretrained_models/model_ir_se50.pth',
	'resnet34': 'pretrained_models/resnet34-333f7ec4.pth',
	# stylegan2 generators
	'stylegan': 'pretrained_models/stylegan2-config-f.pt',
	'stylegan_ada_wild': 'pretrained_models/afhqwild.pt',

	# WEncoders for training on various domains
	'faces_encoder': 'pretrained_models/faces_encoder.pt',
}

edit_paths = {

	'edit': {
		'delta_i_c': 'editing/edit/sketchedit/style/fs.npy',
		's_statistics': 'editing/edit/sketchedit/style/std_mean',
		'templates': 'editing/edit/sketchedit/style/temp.txt'
	}
}