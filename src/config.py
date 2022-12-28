from os.path import join, dirname, abspath
from src import tool

# Basic project info
AUTHOR = "Yzj"
PROGRAM = "Deep Learning-based Smartphone App"
DESCRIPTION = "Biliary atresia diagnose. " \
              "If you find any bug, please new issue. "

# Main CMDs. This decides what kind of cmd you will use.
cmd_list = ['temp', 'train', 'test']

log_name = 'Ultrasonic'

# add parsers to this procedure
globals().update(vars(tool.gen_parser()))


def init_path_config(main_file):
    # global_variables
    gv = globals()
    project_dir = abspath(join(dirname(main_file), '..'))
    gv['project_dir'] = project_dir
    gv['data_dir'] = data_dir = join(project_dir, 'data')
    gv['log_dir'] = join(data_dir, 'log')
    gv['loss_dir'] = join(data_dir, 'loss')
    gv['model_dir'] = join(data_dir, 'model')
    gv['best_model_dir'] = join(data_dir, 'best_model','MPaddOri_AllData_NewModel')  # 载入模型的路径
    gv['step_model_dir'] = join(data_dir, 'EarlyStop_model')
    gv['tb_dir'] = join(data_dir, 'tb')

    #TODO Dataset Path
    gv['ImageNet100_dir'] = ''
    gv['CIFAR10_dir'] = ''


    gv['Gallbladder_train_dir'] = 'Train_dataset_photosPath'
    gv['Gallbladder_test_dir'] = 'Test_dataset_photosPath'


