"""Recall over separate query and candidate sets."""

import torch
import torch.nn.functional as F


def retrieval_recall(image_embeddings, caption_embeddings, image_ids, caption_ids,
                     caption_owners, image_queries=None, caption_queries=None):
    image_ids, caption_ids = list(image_ids), list(caption_ids)
    image_queries = image_ids if image_queries is None else list(image_queries)
    caption_queries = caption_ids if caption_queries is None else list(caption_queries)
    if not all((image_ids, caption_ids, image_queries, caption_queries)):
        raise ValueError('Recall requires nonempty candidates and queries in both directions')
    if (image_embeddings.ndim != 2 or caption_embeddings.ndim != 2
            or image_embeddings.shape[1] != caption_embeddings.shape[1]
            or len(image_embeddings) != len(image_ids) or len(caption_embeddings) != len(caption_ids)):
        raise ValueError('Recall embeddings do not align with candidate IDs')
    image_positions = {item: pos for pos, item in enumerate(image_ids)}
    caption_positions = {item: pos for pos, item in enumerate(caption_ids)}
    if len(image_positions) != len(image_ids) or len(caption_positions) != len(caption_ids):
        raise ValueError('Recall candidate IDs must be unique')
    try:
        owners = [image_positions[caption_owners[item]] for item in caption_ids]
        iq = [image_positions[item] for item in image_queries]
        cq = [caption_positions[item] for item in caption_queries]
    except KeyError as exc:
        raise ValueError(f'Recall queries or caption owners do not align with candidates: {exc}') from exc
    if not set(iq).issubset(owners):
        raise ValueError('An image query has no ground-truth caption in the candidate pool')

    ie = F.normalize(image_embeddings, dim=-1)
    te = F.normalize(caption_embeddings, dim=-1)
    owner_indices = torch.tensor(owners, device=ie.device)
    i_rank = (ie[iq] @ te.T).topk(min(10, len(te)), dim=1).indices
    t_rank = (te[cq] @ ie.T).topk(min(10, len(ie)), dim=1).indices
    image_hits = owner_indices[i_rank].eq(torch.tensor(iq, device=ie.device)[:, None])
    caption_hits = t_rank.eq(owner_indices[cq, None])
    metrics = {}
    for direction, hits in [('i2t', image_hits), ('t2i', caption_hits)]:
        recalls = [hits[:, :k].any(dim=1).double().mean().item() for k in (1, 5, 10)]
        metrics.update({f'{direction}_R@{k}': score for k, score in zip((1, 5, 10), recalls)})
        metrics[f'{direction}_score'] = sum(recalls) / len(recalls)
    metrics['mean_score'] = (metrics['i2t_score'] + metrics['t2i_score']) / 2
    return metrics
