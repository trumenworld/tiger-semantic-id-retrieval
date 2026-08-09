"""Compare legacy and prefix-table generation with one fixed checkpoint."""

import argparse
import copy
import json
import time
from functools import partial

import torch
from torch.utils.data import DataLoader, Subset

from modeling import utils
from modeling.dataloader import BatchProcessor
from modeling.dataset import Dataset
from modeling.models import CorrectItemsLogitsProcessor, TigerModel
from modeling.models.prefix_logits import PrefixTableLogitsProcessor
from modeling.utils import create_masked_tensor, fix_random_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--users', type=int, default=512)
    return parser.parse_args()


def build_model(config, processor):
    num_codebooks = config['dataset']['num_codebooks']
    model = TigerModel(
        embedding_dim=config['model']['embedding_dim'],
        codebook_size=config['model']['codebook_size'],
        sem_id_len=num_codebooks,
        user_ids_count=config['model']['user_ids_count'],
        num_positions=config['model']['num_positions'],
        num_heads=config['model']['num_heads'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        num_beams=config['model']['num_beams'],
        num_return_sequences=config['model']['top_k'],
        activation=config['model']['activation'],
        d_kv=config['model']['d_kv'],
        dropout=config['model']['dropout'],
        layer_norm_eps=config['model']['layer_norm_eps'],
        initializer_range=config['model']['initializer_range'],
        logits_processor=partial(
            processor,
            num_codebooks,
            config['model']['codebook_size'],
            config['dataset']['index_json_path'],
            config['model']['num_beams'],
        ),
    ).to(utils.DEVICE)
    return model.eval()


def load_weights(path):
    state = torch.load(path, map_location=utils.DEVICE, weights_only=False)
    if 'best_checkpoint' in state and state['best_checkpoint'] is not None:
        return state['best_checkpoint']
    if 'model' in state:
        return state['model']
    return state


def run(model, dataloader):
    predictions = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in dataloader:
            for key, value in batch.items():
                batch[key] = value.to(utils.DEVICE)
            predictions.append(model(batch)['predictions'].cpu())
    if utils.DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    return torch.cat(predictions), time.perf_counter() - started


def main():
    args = parse_args()
    fix_random_seed(42)
    with open(args.params, 'r', encoding='utf-8') as handle:
        config = json.load(handle)

    dataset = Dataset.create(
        inter_json_path=config['dataset']['inter_json_path'],
        max_sequence_length=config['dataset']['max_sequence_length'],
        sampler_type=config['dataset']['sampler_type'],
        is_extended=True,
    )
    _, validation_sampler, _ = dataset.get_samplers()
    sample_count = min(args.users, len(validation_sampler))
    subset = Subset(validation_sampler, range(sample_count))
    processor = BatchProcessor.create(
        config['dataset']['index_json_path'],
        config['dataset']['num_codebooks'],
        config['model']['user_ids_count'],
    )
    dataloader = DataLoader(
        subset,
        batch_size=config['dataloader']['validation_batch_size'],
        shuffle=False,
        drop_last=False,
        collate_fn=processor,
    )
    weights = load_weights(args.checkpoint)

    legacy = build_model(config, CorrectItemsLogitsProcessor)
    legacy.load_state_dict(weights)
    legacy_predictions, legacy_seconds = run(legacy, dataloader)
    del legacy
    if utils.DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    prefix = build_model(config, PrefixTableLogitsProcessor)
    prefix.load_state_dict(weights)
    prefix_predictions, prefix_seconds = run(prefix, dataloader)

    exact = torch.equal(legacy_predictions, prefix_predictions)
    mismatches = (legacy_predictions != prefix_predictions).sum().item()
    print('users:', sample_count)
    print('exact predictions:', exact)
    print('token mismatches:', mismatches)
    print('legacy seconds:', round(legacy_seconds, 3))
    print('prefix-table seconds:', round(prefix_seconds, 3))
    print('end-to-end speedup:', round(legacy_seconds / prefix_seconds, 2), 'x')
    if not exact:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
