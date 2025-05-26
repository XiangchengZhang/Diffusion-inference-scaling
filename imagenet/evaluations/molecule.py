import pickle
import torch
import torch.nn as nn
from tqdm import tqdm
from .base import BaseEvaluator
from tasks.networks.egnn.EGNN_prop import EGNN
from diffusion.ddim import MoleculeSampler
from tasks.networks.qm9 import dataset
from tasks.networks.qm9.utils import compute_mean_mad
from tasks.networks.qm9.analyze import analyze_stability_for_molecules
from tasks.networks.qm9.datasets_config import get_dataset_info
import logger

class MoleculeEvaluator(BaseEvaluator):

    def __init__(self, args):
        super(MoleculeEvaluator, self).__init__()

        self.args = args

        with open(self.args.args_classifiers_path, 'rb') as f:
            args_classifier = pickle.load(f)
        args_classifier.device = self.args.device
        args_classifier.model_name = 'egnn'

        classifier = EGNN(in_node_nf=5, in_edge_nf=0, hidden_nf=args_classifier.nf,
                          device=args_classifier.device, n_layers=args_classifier.n_layers,
                          coords_weight=1.0, attention=args_classifier.attention,
                          node_attr=args_classifier.node_attr)
        classifier_state_dict = torch.load(self.args.classifiers_path, map_location=torch.device('cpu'))
        classifier.load_state_dict(classifier_state_dict)
        self.model = classifier

        args.args_gen.load_charges = False
        dataloaders = self._get_dataloader(args.args_gen)
        property_norms = compute_mean_mad(dataloaders, [args.target], args.args_gen.dataset)
        self.mean, self.mad = property_norms[args.target]['mean'], property_norms[args.target]['mad']
        self.dataset_info = get_dataset_info(args.args_gen.dataset, args.args_gen.remove_h)


    @staticmethod
    def _get_dataloader(args_gen):
        dataloaders, charge_scale = dataset.retrieve_dataloaders(args_gen)
        return dataloaders
    @staticmethod
    def _get_adj_matrix(n_nodes, batch_size, device):
        rows, cols = [], []
        for batch_idx in range(batch_size):
            for i in range(n_nodes):
                for j in range(n_nodes):
                    rows.append(i + batch_idx * n_nodes)
                    cols.append(j + batch_idx * n_nodes)

        edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
        return edges

    @torch.no_grad()
    def _compute_metrics(self, samples):  # one_hot, charges, x, node_mask
        self.model.eval()
        one_hot, charges, x, node_mask, target = samples
        molecules = {'one_hot': [], 'x': [], 'node_mask': []}
        count = 0
        context = []
        tot_num = one_hot.size(0)
        device = self.args.device
        loss_fn = nn.L1Loss(reduction='none')
        all_loss = []

        for bs in tqdm(range(0, tot_num, self.args.eval_batch_size), desc='Computing MAE',
                       total=tot_num // self.args.eval_batch_size):
            cur_slice = slice(bs, min(bs + self.args.eval_batch_size, tot_num))

            molecules['one_hot'].append(one_hot[cur_slice].detach().cpu())
            molecules['x'].append(x[cur_slice].detach().cpu())
            molecules['node_mask'].append(node_mask[cur_slice].detach().cpu())

            label = target[cur_slice].to(device).to(torch.float32)

            nodes = one_hot[cur_slice].to(device).to(torch.float32)
            atom_positions = x[cur_slice].to(device).to(torch.float32)
            
            batch_size, n_nodes, _ = atom_positions.size()
            
            nodes = nodes.view(batch_size * n_nodes, -1)
            atom_positions = atom_positions.view(batch_size * n_nodes, -1)
            
            atom_mask = node_mask[cur_slice].to(device).to(torch.float32)

            edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
            diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0).unsqueeze(-1).to(device)
            edge_mask *= diag_mask
            edge_mask = edge_mask.view(batch_size * n_nodes * n_nodes, 1).to(device)
                        
            atom_mask = atom_mask.view(batch_size * n_nodes, 1)
            edges = self._get_adj_matrix(n_nodes, batch_size, device)
            context.append(label)

            pred = self.model(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask,
                              edge_mask=edge_mask, n_nodes=n_nodes)
            loss = loss_fn(self.mad * pred + self.mean, label.squeeze())

            all_loss.append(loss)

        all_loss = torch.cat(all_loss, dim=0)
        # torch.save(context, os.path.join(result_path, 'context.pt'))
        molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
        stability_dict, rdkit_metrics = analyze_stability_for_molecules(
            molecules, self.dataset_info, rdkit=True)
        mae = all_loss.mean().item()
        return mae, stability_dict, rdkit_metrics

    def evaluate(self, samples):
        samples = MoleculeSampler.obj_to_tensor(samples)

        logger.log(f"Evaluating {len(samples[0])} samples")

        mae, stability_dict, rdkit_metrics = self._compute_metrics(samples)
        metrics = {
            'mae': mae,
            'molecule_stability': stability_dict['mol_stable'],
            'atom_stability': stability_dict['atm_stable'],
            'validity': rdkit_metrics[0][0],
            'uniqueness': rdkit_metrics[0][1],
            'novelty': rdkit_metrics[0][2],
            'nan_rate': 1 - len(samples[0]) / self.args.num_samples,
        }

        return metrics
