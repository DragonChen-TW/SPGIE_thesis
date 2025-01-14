# ========== LIME Graph ==========
from lime import lime_base
from lime.wrappers.scikit_image import SegmentationAlgorithm
from torch_geometric.data import Batch

# # lime_image
import copy
from functools import partial
import sklearn
from sklearn.utils import check_random_state
from skimage.color import gray2rgb
from tqdm.auto import tqdm
# # lime_base
import numpy as np
import torch

def draw_graph(graph):
    pos = torch.flip(graph.pos, dims=(1,)).detach().cpu().numpy()
    x = (graph.x[:, 2:] * 255).int()
    util.draw_superpixel_from_graph(pos, x, graph, multi_graph=False)

class GraphExplanation:
    def __init__(self, image, graph, segments):
        """Init
        
        Args:
            image: 2d np.array
            graph: torch_geometric.data.Data
            segments: 2d np array, with the output from skimage.segmentation
        """
        self.image = image
        self.graph = graph
        self.segments = segments
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = {}
        self.score = {}
    
    def get_image_and_mask(self,
                           label, positive_only=True,
                           negative_only=False, hide_rest=False,
                           num_features=5, min_weight=0.
        ):
        """Init function.

        Args:
            label: label to explain
            positive_only: if True, only take superpixels that positively contribute to
                the prediction of the label.
            negative_only: if True, only take superpixels that negatively contribute to
                the prediction of the label. If false, and so is positive_only, then both
                negativey and positively contributions will be taken.
                Both can't be True at the same time
            hide_rest: if True, make the non-explanation part of the return
                image gray
            num_features: number of superpixels to include in explanation
            min_weight: minimum weight of the superpixels to include in explanation

        Returns:
            (image, mask), where image is a 3d numpy array and mask is a 2d
            numpy array that can be used with
            skimage.segmentation.mark_boundaries
        """
        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        if positive_only & negative_only:
            raise ValueError("Positive_only and negative_only cannot be true at the same time.")
        segments = self.segments
        image = self.image
        exp = self.local_exp[label]
        mask = np.zeros(segments.shape, segments.dtype)
        if hide_rest:
            temp = np.zeros(self.image.shape)
        else:
            temp = self.image.copy()
        if positive_only:
            fs = [x[0] for x in exp
                  if x[1] > 0 and x[1] > min_weight][:num_features]
        if negative_only:
            fs = [x[0] for x in exp
                  if x[1] < 0 and abs(x[1]) > min_weight][:num_features]
        if positive_only or negative_only:
            for f in fs:
                temp[segments == f] = image[segments == f].copy()
                mask[segments == f] = 1
            return temp, mask
        else:
            for f, w in exp[:num_features]:
                if np.abs(w) < min_weight:
                    continue
                c = 0 if w < 0 else 1
                mask[segments == f] = -1 if w < 0 else 1
                temp[segments == f] = image[segments == f].copy()
                temp[segments == f, c] = np.max(image)
            return temp, mask
        
class LimeGraphExplainer:
    """Explains predictions on Graph (torch_geometric.data.Data there) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self, kernel_width=.25, kernel=None, verbose=False,
                 feature_selection='auto', random_state=None):
        """Init function.

        Args:
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        self.verbose = verbose
        kernel_width = float(kernel_width)

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(kernel_fn, verbose, random_state=self.random_state)
        
    
    def explain_instance(self, image, graph, node2map, classifier_fn, labels=(1,),
                         hide_color=None,
                         top_labels=5, num_features=100000, num_samples=1000,
                         batch_size=10,
                         segmentation_fn=None,
                         segments=None,
                         distance_metric='cosine',
                         model_regressor=None,
                         random_seed=None,
                         progress_bar=True,
                        ):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            image: 3 dimension RGB image. If this is only two dimensional,
                we will assume it's a grayscale image and call gray2rgb.
            graph: torch_geometric's graph.
            node2map: Dict. We assume the give graph has been segmented before. node2map is the mapping of superpixels to original regions of piexls. 
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            hide_color: If not None, will hide superpixels with this color.
                Otherwise, use the mean pixel color of the image.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            batch_size: batch size for model predictions
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
            random_seed: integer used as random seed for the segmentation
                algorithm. If None, a random integer, between 0 and 1000,
                will be generated using the internal random number generator.
            progress_bar: if True, show tqdm progress bar.

        Returns:
            An GraphExplanation object with the corresponding
            explanations.
        """
        if len(image.shape) == 2:
            image = gray2rgb(image)
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        if segmentation_fn is None and segments is not None:
            segmentation_fn = SegmentationAlgorithm('slic',
                                  n_segments=75,
                                  multichannel=True,
                                  slic_zero=True, start_label=0
                              )
        if segments is None:
            segments = segmentation_fn(image)

        

#         fudged_image = image.copy()
#         if hide_color is None:
#             for x in np.unique(segments):
#                 fudged_image[segments == x] = (
#                     np.mean(image[segments == x][:, 0]),
#                     np.mean(image[segments == x][:, 1]),
#                     np.mean(image[segments == x][:, 2]))
#         else:
#             fudged_image[:] = hide_color

        top = labels

        data, labels = self.data_labels(image, graph, segments,
                                        classifier_fn, num_samples,
                                        batch_size=batch_size,
                                        progress_bar=progress_bar)

        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        ret_exp = GraphExplanation(image, graph, segments)
        if top_labels:
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()
        for label in top:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                data, labels, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp

    def data_labels(self,
                    image,
                    graph,
                    segments,
                    classifier_fn,
                    num_samples,
                    batch_size=10,
                    progress_bar=True):
        """Generates images and predictions in the neighborhood of this image.

        Args:
            image: 3d numpy array, the image
            fudged_image: 3d numpy array, image to replace original image when
                superpixel is turned off
            segments: segmentation of the image
            classifier_fn: function that takes a list of images and returns a
                matrix of prediction probabilities
            num_samples: size of the neighborhood to learn the linear model
            batch_size: classifier_fn will be called on batches of this size.
            progress_bar: if True, show tqdm progress bar.

        Returns:
            A tuple (data, labels), where:
                data: dense num_samples * num_superpixels
                labels: prediction probabilities matrix
        """
        n_features = np.unique(segments).shape[0]
        shuffled = np.random.rand(num_samples, n_features)
        shuffled[0, :] = 0
        shuffled = shuffled < 0.7
        
        graphs = []
        labels = []
        rows = tqdm(shuffled) if progress_bar else shuffled
        verbose = self.verbose
        for shuf in rows:
            temp = copy.deepcopy(graph)
            
            mask = torch.zeros(temp.edge_index.shape[1], dtype=bool)
            shuf_idxs = np.arange(shuf.shape[0])[shuf]
            edge_index = temp.edge_index.clone().detach().cpu()
            for s in shuf_idxs:
                mask = mask | ((edge_index == s).sum(dim=0) != 0)
            mask = ~(mask.bool())
            
            temp.pos = temp.pos[shuf, :]
            temp.x = temp.x[shuf, :]
            
            #     update number of edge_index
            e_index = temp.edge_index[:, mask]
            counter = 0
            for s in shuf_idxs:
                e_index[e_index == s] = counter
                counter += 1
            temp.edge_index = e_index
            
            # Batch inference
            graphs.append(temp)
            if len(graphs) == batch_size:
                batch = Batch.from_data_list(graphs)
                preds = classifier_fn(batch).squeeze()
                labels.extend(preds)
                if verbose:
                    verbose = False
                    for i, g in enumerate(graphs):
                        draw_graph(g)
                        print(preds[i].argmax(axis=0), preds[i])
                graphs = []
        if len(graphs) > 0:
            batch = Batch.from_data_list(graphs)
            preds = classifier_fn(batch)
            labels.extend(preds)

#             # instance-by-instance inference
#             temp.batch = torch.zeros(temp.x.shape[0], dtype=int).to(device)
            
            pred = classifier_fn(temp)
            labels.append(pred.squeeze())
            
#             if verbose:
#                 draw_graph(temp)
#                 print('pred', graph.y, pred.argmax(axis=0).item(), [round(p, 4) for p in pred.squeeze().tolist()])
        
        
        return shuffled, np.array(labels)

# ========== LIME Graph ==========