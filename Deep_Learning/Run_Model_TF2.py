__author__ = 'Brian M Anderson'
# Created on 1/18/2020
from Base_Deeplearning_Code.Data_Generators.Return_Paths import *
import tensorflow as tf
from Base_Deeplearning_Code.Callbacks.TF2_Callbacks import Add_Images_and_LR, SparseCategoricalMeanDSC
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
from Base_Deeplearning_Code.Data_Generators.Return_Paths import Path_Return_Class
from Base_Deeplearning_Code.Models.TF_Keras_Models import my_UNet
from Base_Deeplearning_Code.Cyclical_Learning_Rate.clr_callback_TF2 import CyclicLR
from Return_Train_Validation_Generators_TF2 import return_generators, get_layers_dict, return_base_dict,\
    return_hparams, return_dictionary, return_pandas_df, return_current_df, np, pd
from tensorboard.plugins.hparams.keras import Callback


def run_model(trial_id, min_lr=1e-4, max_lr=1e-2, layers_dict=None, epochs=1000,validation_generator=None,step_size=None,
              paths_class=None, step_size_factor=5, train_generator=None, morfeus_drive='',base_path='', save_a_model=True,
              skip_cyclic_lr=False, scale_mode='linear_cycle', optimizer='SGD', hparams=None, **kwargs):
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
    checkpoint = ModelCheckpoint(model_path_out, monitor='val_sparse_categorical_mean_dsc',
                                 save_freq='epoch', save_best_only=False, save_weights_only=False, mode='max',
                                 verbose=1)
    tensorboard = TensorBoard(log_dir=tensorboard_output, profile_batch=0)
    lrate = CyclicLR(base_lr=min_lr, max_lr=max_lr, step_size=step_size, step_size_factor=step_size_factor,
                     mode='triangular2', pre_cycle=0, base_reduce_factor=2, scale_mode=scale_mode,
                     step_size_factor_scale=lambda x: x)
    # add_images = Add_Images_and_LR(log_dir=tensorboard_output, add_images=False)
    callbacks = [tensorboard]#, add_images]
    if hparams is not None:
        hp_callback = Callback(tensorboard_output, hparams=hparams, trial_id='Trial_ID:{}'.format(trial_id))
        callbacks += [hp_callback]
    if not skip_cyclic_lr:
        callbacks += [lrate]
    if save_a_model:
        callbacks += [checkpoint]
    model = my_UNet(layers_dict=layers_dict, image_size=(None, None, None, 1), mask_output=True)
    Model_val = model.created_model
    print('\n\n\n\nRunning {}\n\n\n\n'.format(tensorboard_output))
    Model_val.compile(optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy(), SparseCategoricalMeanDSC(num_classes=2)])
    Model_val.fit(train_generator.data_set, epochs=epochs, callbacks=callbacks,
                  validation_data=validation_generator.data_set, validation_steps=len(validation_generator),
                  steps_per_epoch=len(train_generator))


def train_model(epochs=None,bn_before_activation=True, save_a_model=False, batch_size=16,model_name = '3D_Fully_Atrous',
                step_size_factor=8, optimizer='SGD'):

    base_path, morfeus_drive, train_generator, validation_generator = return_generators(batch_size=batch_size)
    print(base_path)


    excel_path = os.path.join(morfeus_drive,'parameters_list_by_trial_id.xlsx')
    num_cycles = 10
    step_size = len(train_generator)
    base_dict = return_base_dict(step_size_factor=step_size_factor,
                                 save_a_model=save_a_model, optimizer=optimizer)
    if epochs is None:
        epochs = step_size_factor
        for _ in range(1,num_cycles):
            epochs += (step_size_factor * 2)
        epochs += 2
        epochs = min([1000,epochs])
        epochs = max([300, epochs])
    epochs = 5
    for iteration in range(3):
        overall_dictionary = return_dictionary(base_dict, optimizer=optimizer)
        for run_data in overall_dictionary:
            run_data['Iteration'] = iteration
            data_frame = return_pandas_df(excel_path, features_list=list(run_data.keys()))
            trial_id = len(data_frame['Trial_ID'])
            run_data['Trial_ID'] = trial_id
            current_run_df, features_list = return_current_df(run_data, features_list=data_frame.columns)
            if np.min(current_run_df[features_list].isin(data_frame[features_list]).values,axis=1).any():
                print('Already done')
                continue
            data_frame = data_frame.append(current_run_df, ignore_index=True)
            data_frame.to_excel(excel_path, index=0)

            hparams = return_hparams(run_data, features_list=features_list)

            layers_dict = get_layers_dict(**run_data, bn_before_activation=bn_before_activation)
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
                      validation_generator=validation_generator,save_a_model=save_a_model,
                      paths_class=paths_class,morfeus_drive=morfeus_drive, hparams=hparams,
                      base_path=base_path, epochs=epochs,**run_data)


if __name__ == '__main__':
    pass
