__author__ = 'Brian M Anderson'
# Created on 1/18/2020
from Base_Deeplearning_Code.Data_Generators.Return_Paths import *
import tensorflow as tf
from Base_Deeplearning_Code.Callbacks.TF2_Callbacks import Add_Images_and_LR, SparseCategoricalMeanDSC
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
from Base_Deeplearning_Code.Data_Generators.Return_Paths import Path_Return_Class
from Base_Deeplearning_Code.Models.TF_Keras_Models import my_UNet
from Base_Deeplearning_Code.Cyclical_Learning_Rate.clr_callback_TF2 import CyclicLR
from Return_Train_Validation_Generators_TF2 import return_generators, return_base_dict_dense, get_layers_dict, return_base_dict,\
    return_hparams, return_dictionary, return_pandas_df, return_current_df, np, return_paths, return_best_dictionary, get_layers_dict_new, get_layers_dict_dense, return_dictionary_dense
from tensorboard.plugins.hparams.keras import Callback


def run_model(trial_id, min_lr=1e-4, max_lr=1e-2, layers_dict=None, epochs=1000,validation_generator=None,step_size=None,
              paths_class=None, step_size_factor=5, train_generator=None, morfeus_drive='',base_path='', run_best=False,
              skip_cyclic_lr=False, scale_mode='linear_cycle', optimizer='SGD', hparams=None,concat=True, **kwargs):
    if step_size is None:
        step_size = len(train_generator)
    if not os.path.exists(morfeus_drive):
        print('Morf wrong')
        return None
    if not os.path.exists(base_path):
        print('base wrong')
        return None


    model_path_out = paths_class.model_path_out
    tensorboard_output = paths_class.tensorboard_path_out
    if optimizer == 'SGD':
        optimizer = tf.keras.optimizers.SGD()
    elif optimizer == 'Adam':
        optimizer = tf.keras.optimizers.Adam()
    optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
    print('Learning rate is {}'.format(min_lr))

    if os.listdir(tensorboard_output):
        print('already done')
        return None
    checkpoint_path = os.path.join(model_path_out,'cp-best.ckpt')
    image_frequency = 20
    val_frequency = 5
    if run_best:
        image_frequency = 10
        val_frequency = 1
        checkpoint_path = os.path.join(model_path_out,'cp-{epoch:04d}.ckpt')
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss',
                                 save_freq='epoch', save_best_only=False, save_weights_only=True, mode='min',
                                 verbose=1)
    tensorboard = TensorBoard(log_dir=tensorboard_output, profile_batch='300,401', histogram_freq=5, write_graph=True)
    lrate = CyclicLR(base_lr=min_lr, max_lr=max_lr, step_size=step_size, step_size_factor=step_size_factor,
                     mode='triangular2', pre_cycle=0, base_reduce_factor=5, scale_mode=scale_mode,
                     step_size_factor_scale=lambda x: x + 2, reduction_factor=5)
    add_images = Add_Images_and_LR(log_dir=tensorboard_output, validation_data=validation_generator.data_set,
                                   number_of_images=len(validation_generator), add_images=True, image_frequency=image_frequency,
                                   threshold_x=True)
    callbacks = [tensorboard, add_images]
    if hparams is not None:
        hp_callback = Callback(tensorboard_output, hparams=hparams, trial_id='Trial_ID:{}'.format(trial_id))
        callbacks += [hp_callback]
    if not skip_cyclic_lr:
        callbacks += [lrate]
    callbacks += [checkpoint]
    if not run_best:
        callbacks += [EarlyStopping(patience=15, verbose=1)]
    model = my_UNet(layers_dict=layers_dict, image_size=(None, None, None, 1), mask_output=True, concat_not_add=concat)
    Model_val = model.created_model
    print('\n\n\n\nRunning {}\n\n\n\n'.format(tensorboard_output))
    Model_val.compile(optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy(), SparseCategoricalMeanDSC(num_classes=2)])
    Model_val.fit(train_generator.data_set, epochs=epochs, callbacks=callbacks, steps_per_epoch=step_size,
                  validation_data=validation_generator.data_set, validation_steps=len(validation_generator),
                  validation_freq=val_frequency)
    tf.keras.backend.clear_session()
    return None


def compare_base_current(data_frame, current_run_df, features_list):
    current_array = current_run_df[features_list].values
    base_array = data_frame[features_list].values
    if np.any(base_array) and np.max([np.min(i == current_array) for i in base_array]):
        return True
    return False


def train_model(epochs=None, save_a_model=False, model_name='3D_Fully_Atrous',
                run_best=False, debug=False, add='', dense=False):
    batch_size = 16
    if add != '':
        batch_size = 8
    optimizers = ['Adam']
    concat = True
    if run_best:
        save_a_model = True
    bn_before_activation = True
    step_size_factor = 6
    threshold = True
    for iteration in range(3):
        for flip in [False]:
            for threshold_val in [10]:
                for change_background in [True, False]:
                    for optimizer in optimizers:
                        cache_add = ''
                        if change_background:
                            cache_add += '_change_bckgrd'
                        base_path, morfeus_drive = return_paths()
                        # base_dict = return_base_dict(step_size_factor=step_size_factor,
                        #                              save_a_model=save_a_model, optimizer=optimizer)
                        base_dict = return_base_dict_dense(step_size_factor=step_size_factor, save_a_model=save_a_model)
                        if run_best:
                            if dense:
                                overall_dictionary = return_dictionary_dense(base_dict,run_best=run_best)
                            else:
                                overall_dictionary = return_best_dictionary(base_dict)
                        else:
                            # overall_dictionary = return_dictionary(base_dict)
                            overall_dictionary = return_dictionary_dense(base_dict)
                        overall_dictionary = np.asarray(overall_dictionary)
                        perm = np.arange(len(overall_dictionary))
                        np.random.shuffle(perm)
                        overall_dictionary = overall_dictionary[perm]
                        if debug:
                            i = 0
                        for run_data in overall_dictionary:
                            run_data['percentile_normed'] = True
                            run_data['mirror_max'] = False
                            run_data['Model_Style'] = 'new'
                            run_data['concat'] = concat
                            run_data['flipped'] = flip
                            run_data['change_background'] = change_background
                            run_data['threshold'] = threshold
                            run_data['threshold_val'] = threshold_val
                            if debug:
                                layers_dict = get_layers_dict_new(**run_data, bn_before_activation=bn_before_activation)
                                model = my_UNet(layers_dict=layers_dict, image_size=(None, None, None, 1), mask_output=True,
                                                concat_not_add=True)
                                Model_val = model.created_model
                                Model_val.compile(optimizer,
                                                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                                                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(),
                                                           SparseCategoricalMeanDSC(num_classes=2)])
                                callbacks = []
                                i += 1
                                if os.path.exists(r'D:\Liver_Disease_Ablation\tensorboard\test\{}'.format(i)):
                                    continue
                                k = TensorBoard(log_dir=r'D:\Liver_Disease_Ablation\tensorboard\test\{}'.format(i), profile_batch=0, histogram_freq=5, write_graph=True)
                                k.set_model(Model_val)
                                k.on_train_begin()
                                tf.keras.backend.clear_session()
                                continue
                            if debug:
                                return None
                            tf.random.set_seed(iteration)
                            run_data['batch_size'] = batch_size
                            excel_path = os.path.join(morfeus_drive, 'parameters_list_by_trial_id_Dense.xlsx')
                            print(base_path)
                            run_data['Iteration'] = iteration
                            run_data['Trial_ID'] = 0
                            data_frame = return_pandas_df(excel_path, features_list=list(run_data.keys()))
                            trial_id = 0
                            while trial_id in data_frame['Trial_ID'].values:
                                trial_id += 1
                            run_data['Trial_ID'] = trial_id
                            current_run_df, features_list = return_current_df(run_data, features_list=data_frame.columns)
                            if compare_base_current(data_frame=data_frame, current_run_df=current_run_df, features_list=[i for i in data_frame.columns if i != 'Trial_ID']):
                                print('Already done')
                                continue
                            print(current_run_df)
                            data_frame = data_frame.append(current_run_df, ignore_index=True)
                            data_frame.to_excel(excel_path, index=0)
                            _, _, train_generator, validation_generator = return_generators(batch_size=batch_size, add=add,cache_add=cache_add,
                                                                                            flip=flip, change_background=change_background,
                                                                                            threshold=threshold, threshold_val=threshold_val)
                            step_size = len(train_generator)
                            hparams = return_hparams(run_data, features_list=features_list, excluded_keys=[])

                            # layers_dict = get_layers_dict_new(**run_data, bn_before_activation=bn_before_activation)
                            layers_dict = get_layers_dict_dense(**run_data)
                            paths_class = Path_Return_Class(base_path=base_path, morfeus_path=morfeus_drive, save_model=save_a_model,
                                                            is_keras_model=False)
                            paths_class.define_model_things(model_name, 'Trial_ID_{}'.format(trial_id))
                            tensorboard_output = paths_class.tensorboard_path_out
                            print(tensorboard_output)
                            if os.listdir(tensorboard_output):
                                print('already done')
                                continue
                            run_model(trial_id=str(trial_id), layers_dict=layers_dict, train_generator=train_generator,
                                      step_size=step_size, optimizer=optimizer,
                                      validation_generator=validation_generator,run_best=run_best,
                                      paths_class=paths_class,morfeus_drive=morfeus_drive, hparams=hparams,
                                      base_path=base_path, epochs=epochs,**run_data)
                            return None # break out!


if __name__ == '__main__':
    pass
