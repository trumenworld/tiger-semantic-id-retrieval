import copy
import os
import random

import numpy as np
import torch

from modeling.trainer import MetricCallback, InferenceCallback
from modeling.utils import create_logger, TensorboardWriter, DEVICE

LOGGER = create_logger(name=__name__)


class Trainer:
    def __init__(
            self,
            experiment_name,
            train_dataloader,
            validation_dataloader,
            eval_dataloader,
            model,
            optimizer,
            loss_function,
            ranking_metrics,
            epoch_cnt=None,
            step_cnt=None,
            best_metric=None,
            epochs_threshold=40,
            valid_step=256,
            eval_step=256,
            run_eval_during_training=True,
            checkpoint_every_epochs=1,
            resume_checkpoint_path=None,
            resume_from_latest=False,
            checkpoint_dir='../checkpoints'
    ):
        self._experiment_name = experiment_name
        self._train_dataloader = train_dataloader
        self._validation_dataloader = validation_dataloader
        self._eval_dataloader = eval_dataloader
        self._model = model
        self._optimizer = optimizer
        self._loss_function = loss_function
        self._epoch_cnt = epoch_cnt
        self._step_cnt = step_cnt
        self._best_metric = best_metric
        self._epochs_threshold = epochs_threshold
        self._ranking_metrics = ranking_metrics
        self._run_eval_during_training = run_eval_during_training
        self._checkpoint_every_epochs = checkpoint_every_epochs
        self._resume_checkpoint_path = resume_checkpoint_path
        self._resume_from_latest = resume_from_latest
        self._checkpoint_dir = checkpoint_dir
        os.makedirs(self._checkpoint_dir, exist_ok=True)

        tensorboard_writer = TensorboardWriter(self._experiment_name)

        self._metric_callback = MetricCallback(tensorboard_writer=tensorboard_writer, on_step=1)

        self._validation_callback = InferenceCallback(
            tensorboard_writer=tensorboard_writer,
            step_name='validation',
            model=model,
            dataloader=validation_dataloader,
            on_step=valid_step,
            metrics=ranking_metrics,
            pred_prefix='predictions',
            labels_prefix='labels',
            run_on_step_zero=False,
        )

        self._eval_callback = InferenceCallback(
            tensorboard_writer=tensorboard_writer,
            step_name='eval',
            model=model,
            dataloader=eval_dataloader,
            on_step=eval_step,
            metrics=ranking_metrics,
            pred_prefix='predictions',
            labels_prefix='labels'
        )

    def _save_resume_checkpoint(
            self,
            step_num,
            epoch_num,
            current_metric,
            best_epoch,
            best_checkpoint,
    ):
        if self._resume_checkpoint_path is None:
            return

        state = {
            'model': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'step_num': step_num,
            'epoch_num': epoch_num,
            'current_metric': current_metric,
            'best_epoch': best_epoch,
            'best_checkpoint': best_checkpoint,
            'python_rng_state': random.getstate(),
            'numpy_rng_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()

        temporary_path = f'{self._resume_checkpoint_path}.tmp'
        torch.save(state, temporary_path)
        os.replace(temporary_path, self._resume_checkpoint_path)
        LOGGER.debug(
            'Saved resumable checkpoint at epoch %s, step %s to %s',
            epoch_num,
            step_num,
            self._resume_checkpoint_path,
        )

    def _load_resume_checkpoint(self):
        if (
            not self._resume_from_latest
            or self._resume_checkpoint_path is None
            or not os.path.exists(self._resume_checkpoint_path)
        ):
            return None

        state = torch.load(
            self._resume_checkpoint_path,
            map_location=DEVICE,
            weights_only=False,
        )
        self._model.load_state_dict(state['model'])
        self._optimizer.load_state_dict(state['optimizer'])
        random.setstate(state['python_rng_state'])
        np.random.set_state(state['numpy_rng_state'])
        torch.set_rng_state(state['torch_rng_state'].detach().cpu().clone())
        if torch.cuda.is_available() and 'cuda_rng_state_all' in state:
            torch.cuda.set_rng_state_all([
                rng_state.detach().cpu().clone()
                for rng_state in state['cuda_rng_state_all']
            ])
        LOGGER.debug(
            'Resumed checkpoint from epoch %s, step %s',
            state['epoch_num'],
            state['step_num'],
        )
        return state

    def train(self):
        resume_state = self._load_resume_checkpoint()
        step_num = 0 if resume_state is None else resume_state['step_num']
        epoch_num = 0 if resume_state is None else resume_state['epoch_num']
        current_metric = 0 if resume_state is None else resume_state['current_metric']
        best_epoch = 0 if resume_state is None else resume_state['best_epoch']
        best_checkpoint = None if resume_state is None else resume_state['best_checkpoint']

        LOGGER.debug('Start training...')

        while (step_num < 200_000):
            if best_epoch + self._epochs_threshold < epoch_num:
                LOGGER.debug(
                    'There is no progress during {} epochs. Finish training'.format(self._epochs_threshold))
                break

            LOGGER.debug(f'Start epoch {epoch_num}')
            for batch in self._train_dataloader:
                self._model.train()

                # Move to device
                for key, values in batch.items():
                    batch[key] = values.to(DEVICE)

                # Forward step
                batch.update(self._model(batch))
                loss = self._loss_function(batch)

                # Backward step
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                # Callbacks
                validation_metrics = self._validation_callback(step_num)
                evaluation_metrics = (
                    self._eval_callback(step_num)
                    if self._run_eval_during_training
                    else {}
                )

                # Log metrics
                self._metric_callback(key='loss', value=loss.item(), step_num=step_num, prefix='train')
                for key, value in validation_metrics.items():
                    self._metric_callback(key=key, value=value, step_num=step_num, prefix='validation')
                for key, value in evaluation_metrics.items():
                    self._metric_callback(key=key, value=value, step_num=step_num, prefix='eval')

                # Update best checkpoint
                if self._best_metric in validation_metrics and (
                    best_checkpoint is None
                    or current_metric <= validation_metrics[self._best_metric]
                ):
                    print(step_num, validation_metrics[self._best_metric], current_metric)
                    current_metric = validation_metrics[self._best_metric]
                    best_checkpoint = copy.deepcopy(self._model.state_dict())
                    best_epoch = epoch_num

                step_num += 1

            epoch_num += 1
            if epoch_num % self._checkpoint_every_epochs == 0:
                self._save_resume_checkpoint(
                    step_num,
                    epoch_num,
                    current_metric,
                    best_epoch,
                    best_checkpoint,
                )
        LOGGER.debug('Training procedure has been finished!')
        return best_checkpoint

    def eval(self):
        evaluation_metrics = self._eval_callback(0)
        for key, value in evaluation_metrics.items():
            print(key, value)

    def save(self):
        LOGGER.debug('Saving model...')
        checkpoint_path = f'{self._checkpoint_dir}/{self._experiment_name}_final_state.pth'
        torch.save(self._model.state_dict(), checkpoint_path)
        LOGGER.debug('Saved model as {}'.format(checkpoint_path))

    def load(self, checkpoint):
        self._model.load_state_dict(checkpoint)
