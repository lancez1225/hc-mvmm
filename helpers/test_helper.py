"""Tester driving forward + decode + (optional) KITTI evaluation."""

import os
import time

import torch
import tqdm

from helpers.checkpoint_helper import load_checkpoint
from utils.decode_utils import decode_detections


class Tester:
    """Loads a checkpoint, runs inference on a split, and (optionally) evaluates."""

    def __init__(self, cfg, model, data_loader, result_dir, logger):
        """Stores config and prepares the model for inference.

        Args:
            cfg: ``tester`` sub-dictionary of the YAML config.
            model: The detector to evaluate.
            data_loader: Validation/test :class:`DataLoader`.
            result_dir: Output directory for the KITTI-format text results.
            logger: Logger used for progress messages.
        """
        self.cfg = cfg
        self.model = model
        self.data_loader = data_loader
        self.class_names = data_loader.dataset.class_names
        self.result_dir = result_dir
        self.logger = logger
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def test(self):
        """Runs inference, optionally saves results and triggers evaluation.

        Environment variables ``MAX_ITERS_FOR_SPEED`` and ``SKIP_EVAL_FOR_SPEED``
        can be set to bound the number of iterations and to skip evaluation
        when measuring inference speed.
        """
        assert os.path.exists(self.cfg['checkpoint'])
        load_checkpoint(
            file_name=self.cfg['checkpoint'],
            model=self.model,
            optimizer=None,
            map_location=self.device,
            logger=self.logger,
        )

        torch.set_grad_enabled(False)
        self.model.eval()
        all_det = {}
        progress_bar = tqdm.tqdm(
            total=len(self.data_loader),
            dynamic_ncols=True, leave=True, desc='batches',
        )

        total_frames = 0
        total_time = 0.0
        max_iters = int(os.environ.get('MAX_ITERS_FOR_SPEED', '0'))

        idx = -1
        for idx, batch_dict in enumerate(self.data_loader):
            batch_dict = self.data_loader.dataset.load_data_to_gpu(batch_dict, self.device)

            torch.cuda.synchronize()
            t_start = time.time()
            batch_dict = self.model(
                batch_dict,
                score_thresh=self.cfg['score_thresh'],
                nms_thresh=self.cfg['nms_thresh'],
            )

            det = decode_detections(batch_dict)
            all_det.update(det)

            torch.cuda.synchronize()
            iter_time = time.time() - t_start
            cur_bs = batch_dict['batch_size'] if 'batch_size' in batch_dict else 1
            total_frames += cur_bs
            total_time += iter_time
            fps_cur = cur_bs / max(iter_time, 1e-6)
            avg_fps = total_frames / max(total_time, 1e-6)

            progress_bar.set_postfix_str(
                f'fps={fps_cur:.2f}, avg={avg_fps:.2f}, it={iter_time * 1000:.1f}ms'
            )
            if (idx + 1) % 50 == 0 or idx == 0:
                self.logger.info(
                    f'[Speed] iter {idx + 1}/{len(self.data_loader)} | '
                    f'fps={fps_cur:.2f} | avg_fps={avg_fps:.2f} | '
                    f'latency={iter_time * 1000:.2f} ms'
                )

            progress_bar.update()

            if max_iters > 0 and (idx + 1) >= max_iters:
                self.logger.info(
                    f'Reached MAX_ITERS_FOR_SPEED={max_iters}, stopping early.'
                )
                break
        progress_bar.close()

        avg_fps = (total_frames / max(total_time, 1e-6)) if total_frames > 0 else 0.0
        avg_latency = (
            (total_time / max(total_frames, 1)) * 1000.0 if total_frames > 0 else 0.0
        )
        self.logger.info(
            f'Overall FPS: {avg_fps:.2f} | '
            f'Avg latency: {avg_latency:.2f} ms/frame | Frames: {total_frames}'
        )

        if os.environ.get('SKIP_EVAL_FOR_SPEED', '0') == '1':
            self.logger.info('Speed-only run: skip saving results and evaluation.')
            return

        self.logger.info('==> Saving results...')
        self.save_results(all_det, self.result_dir)
        self.logger.info('==> Done.')
        self.evaluate(self.result_dir)

    def save_results(self, all_det, result_dir):
        """Writes one KITTI-format text file per frame inside ``result_dir``."""
        os.makedirs(result_dir, exist_ok=True)
        for img_id in all_det.keys():
            output_path = os.path.join(result_dir, '{:06d}.txt'.format(int(img_id)))
            with open(output_path, 'w') as f:
                for row in all_det[img_id]:
                    class_name = self.class_names[int(row[0])]
                    f.write(f'{class_name} 0.0 0')
                    for j in range(1, 14):
                        f.write(' {:.2f}'.format(row[j]))
                    f.write('\n')

    def evaluate(self, result_dir):
        """Calls the dataset's KITTI evaluator if labels are available."""
        dataset = self.data_loader.dataset
        if getattr(dataset, 'split', None) == 'test':
            self.logger.info(
                'Test split detected: skip evaluation and keep exported result files.'
            )
            return

        label_dir = getattr(dataset, 'label_dir', None)
        if label_dir is not None and not os.path.isdir(label_dir):
            self.logger.info(
                'Label directory unavailable: skip evaluation and keep exported result files.'
            )
            return

        dataset.eval(result_dir=result_dir, logger=self.logger)
