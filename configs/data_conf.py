from configs import trans_conf
from configs.path_conf import dataset_paths


DATASETS = {

	'CUHK': {
		'transforms': trans_conf.EncodeTransforms,
		'train_source_root': dataset_paths['CUHK_train_P'],
		'train_target_root': dataset_paths['CUHK_train_S'],
		'test_source_root': dataset_paths['CUHK_test_P'],
		'test_target_root': dataset_paths['CUHK_test_S'],
	},
	'CUFSF': {
		'transforms': trans_conf.EncodeTransforms,
		'train_source_root': dataset_paths['CUFSF_train_P'],
		'train_target_root': dataset_paths['CUFSF_train_S'],
		'test_source_root': dataset_paths['CUFSF_test_P'],
		'test_target_root': dataset_paths['CUFSF_test_S'],
	},
	'FS2K': {
		'transforms': trans_conf.EncodeTransforms,
		'train_source_root': dataset_paths['FS2K_train_P'],
		'train_target_root': dataset_paths['FS2K_train_S'],
		'test_source_root': dataset_paths['FS2K_test_P'],
		'test_target_root': dataset_paths['FS2K_test_S'],
	},
	'WildSketch': {
		'transforms': trans_conf.EncodeTransforms,
		'train_source_root': dataset_paths['WildSketch_train_P'],
		'train_target_root': dataset_paths['WildSketch_train_S'],
		'test_source_root': dataset_paths['WildSketch_test_P'],
		'test_target_root': dataset_paths['WildSketch_test_S'],
	},
}