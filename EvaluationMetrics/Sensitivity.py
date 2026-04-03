"""
Sensitivity Metric for XAI Methods (SHAP, LIME)


Sensitivity measures how stable an XAI explanation is when input data changes slightly.
Think of it like this: If you change one test result by 1%, should your medical diagnosis 
explanation completely change? No! Good explanations should be STABLE.


-----------------------------------------------------------

Key Features:
- Measures XAI explanation stability under small input perturbations
- Computes MAD, cosine similarity, and standard deviation metrics
- Handles multi-class classification scenarios


Metrics Provided:
- Mean Absolute Deviation (MAD): Magnitude of explanation changes
- Cosine Similarity: Alignment between original and perturbed explanations
- Standard Deviation: Variability across multiple perturbations


"""




import numpy as np
from scipy.spatial.distance import cosine
from joblib import Parallel, delayed
import pandas as pd

class XAISensitivityEvaluator:

    def __init__(self, model, perturbation_std=0.01, num_perturbations=50, feature_names=None, random_state=None):
        self.model = model
        self.perturbation_std = perturbation_std
        self.num_perturbations = num_perturbations
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state) if random_state is not None else None

        if feature_names is None:

            if hasattr(model, "feature_names_in_"):
                self.feature_names = list(model.feature_names_in_)
            else:
                raise ValueError("feature_names must be provided if model does not have feature_names_in_")
        else:
            self.feature_names = list(feature_names)

    def _perturb_input(self, instance):
        """Returns perturbed instances as DataFrames with feature names"""
        base = instance.values if hasattr(instance, 'values') else instance
        if isinstance(base, pd.DataFrame):
            base_values = base.values
            columns = base.columns
        else:
            base_values = base
            columns = None
        
        if self._rng is not None:
            noise = self._rng.normal(0, self.perturbation_std, (self.num_perturbations, *base_values.shape))
        else:
            noise = np.random.normal(0, self.perturbation_std, (self.num_perturbations, *base_values.shape))
        perturbations = base_values + noise
        
        # in case return as dataframe if original had columns
        if columns is not None:
            return [pd.DataFrame(p.reshape(1, -1), columns=columns) for p in perturbations]
        return perturbations


    # def _perturb_input(self, instance):
    #     # Vectorized perturbation 
    #     base = instance.values if hasattr(instance, 'values') else instance
    #     perturbations = base + np.random.normal(0, self.perturbation_std, (self.num_perturbations, *base.shape))
    #     return perturbations

    def _calculate_sensitivity(self, original_explanation, perturbed_explanations):
        # Vectorized MAD and safe cosine similarity
        orig = original_explanation.flatten()
        perts = np.array([p.flatten() for p in perturbed_explanations])
        mad_scores = np.mean(np.abs(perts - orig), axis=1)

        def _safe_cosine(a, b, eps=1e-12):
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na < eps and nb < eps:
                return 1.0
            if na < eps or nb < eps:
                return 0.0
            return 1 - cosine(a, b)

        cosine_scores = [_safe_cosine(orig, p) for p in perts]
        return {
            'mean_absolute_deviation': np.mean(mad_scores),
            'cosine_similarity': np.mean(cosine_scores),
            'std_deviation': np.std(mad_scores)
        }
    
    def _predict_proba_with_names(self, x):
        # Always return predictions for a DataFrame with correct columns
        if isinstance(x, np.ndarray):
            return self.model.predict_proba(pd.DataFrame(x, columns=self.feature_names))
        elif isinstance(x, pd.DataFrame):
            return self.model.predict_proba(x)
        else:
            # fallback, try to convert
            return self.model.predict_proba(pd.DataFrame(x, columns=self.feature_names))

    def evaluate_shap(self, instance, explainer, class_idx):
        """
        SHAP sensitivity: extract correct per-class SHAP vector for each perturbed instance.
        """
        n_classes = len(self.model.classes_)
        # Get original SHAP vector
        original_shap = explainer.shap_values(instance)
        orig_vec = self.extract_shap_vector(original_shap, class_idx, n_classes)
        # Defensive check: ensure feature count matches
        if orig_vec.shape[0] != instance.shape[1]:
            raise ValueError(f"SHAP vector length ({orig_vec.shape[0]}) does not match feature count ({instance.shape[1]})")
        # Get perturbed instances
        perturbations = self._perturb_input(instance)
        perturbed_vecs = []
        # # Print original SHAP vector for debugging
        # for i, pert in enumerate(perturbations[:5]):
        #     pert_shap = explainer.shap_values(pert)
        #     pert_vec = self.extract_shap_vector(pert_shap, class_idx, n_classes)
        #     print(f"Original SHAP: {orig_vec[:5]}")
        #     print(f"Perturbed SHAP {i}: {pert_vec[:5]}")
        #     print(f"Diff: {np.abs(orig_vec - pert_vec).sum()}")

        for pert in perturbations:
            pert_shap = explainer.shap_values(pert)
            pert_vec = self.extract_shap_vector(pert_shap, class_idx, n_classes)
            perturbed_vecs.append(pert_vec)
        return self._calculate_sensitivity(orig_vec, perturbed_vecs)
    

        

    # # Helper for SHAP extraction 
    # @staticmethod
    # def extract_shap_vector(original_shap, class_idx, n_classes):
    #     """
    #     Extract the SHAP vector for the correct class from any SHAP output format.
    #     """
    #     if isinstance(original_shap, list):
    #         shap_vec = np.array(original_shap[class_idx]).squeeze()
    #     elif isinstance(original_shap, np.ndarray) and original_shap.ndim == 3:
    #         shap_vec = original_shap[:, :, class_idx].squeeze()
    #     elif isinstance(original_shap, np.ndarray) and original_shap.ndim == 2 and original_shap.shape[1] == n_classes:
    #         shap_vec = original_shap[:, class_idx].squeeze()
    #     elif isinstance(original_shap, np.ndarray) and original_shap.ndim == 1:
    #         shap_vec = original_shap
    #     else:
    #         raise ValueError(f"Unexpected SHAP output shape: {type(original_shap)}, {getattr(original_shap, 'shape', None)}")
    #     return shap_vec

    @staticmethod
    def extract_shap_vector(original_shap, class_idx, n_classes):
        """
        Extract the SHAP vector for the correct class from any SHAP output format.
        Handles:
        - List of arrays: [n_classes][n_samples][n_features]
        - Array: [n_samples, n_features, n_classes] == multi-class
        - Array: [n_samples, n_features] (binary) == 2-class
        """
        # List format [n_classes][n_samples][n_features]
        if isinstance(original_shap, list):
            # Defensive check shape
            arr = np.array(original_shap[class_idx])
            if arr.ndim == 2:
                return arr[0]  # [n_samples, n_features] => take first sample
            elif arr.ndim == 1:
                return arr    # [n_features]
            else:
                raise ValueError(f"Unexpected shape in SHAP list: {arr.shape}")
        # Array format [n_samples, n_features, n_classes]
        elif isinstance(original_shap, np.ndarray) and original_shap.ndim == 3:
            return original_shap[0, :, class_idx]
        # Array format: [n_samples, n_features] (binary)
        elif isinstance(original_shap, np.ndarray) and original_shap.ndim == 2:
            # If n_classes == 2, SHAP sometimes returns [n_samples, n_features]
            return original_shap[0]
        # Array format: [n_features]
        elif isinstance(original_shap, np.ndarray) and original_shap.ndim == 1:
            return original_shap
        else:
            raise ValueError(f"Unexpected SHAP output shape: {type(original_shap)}, {getattr(original_shap, 'shape', None)}")



    def evaluate_lime(self, instance, explainer, class_idx):
        # Ensure instance is 1D for LIME
        if hasattr(instance, 'values') and instance.ndim > 1:
            # pandas DataFrame/Series, take the first row if multi-row
            data_row = instance.values[0].flatten()
        elif isinstance(instance, np.ndarray) and instance.ndim > 1:
            #  numpy array, take the first row if multi-row
            data_row = instance[0].flatten()
        elif hasattr(instance, 'flatten'):
            data_row = instance.flatten()
        else:
            data_row = instance 

        # Explain all features to ensure comprehensive comparison
        num_features_to_explain = len(data_row)

        exp = explainer.explain_instance(
            data_row, 
            self._predict_proba_with_names,  
            top_labels=1, 
            num_features=num_features_to_explain
        )

        # exp = explainer.explain_instance(
        #     data_row, 
        #     self.model.predict_proba, 
        #     top_labels=1, 
        #     num_features=num_features_to_explain
        # )
        
        actual_class_idx = class_idx
        if class_idx not in exp.local_exp:
            if not exp.local_exp: # Should not happen if explain instance worked
                    return {'mean_absolute_deviation': np.nan, 'cosine_similarity': np.nan, 'std_deviation': np.nan}
            actual_class_idx = list(exp.local_exp.keys())[0] # Fallback to the first available class
            
        original_weights = dict(exp.as_list(label=actual_class_idx))
        
        perturbations = self._perturb_input(instance) 

        def get_single_lime_explanation(p_instance_1d, model_predict_proba_fn, lime_explainer_obj, target_class_idx, num_feats):
            try:
                # exp_p = lime_explainer_obj.explain_instance(
                #     p_instance_1d, 
                #     model_predict_proba_fn, 
                #     top_labels=1, 
                #     num_features=num_feats
                # )
                
                exp_p = lime_explainer_obj.explain_instance(
                    p_instance_1d, 
                    self._predict_proba_with_names,  
                    top_labels=1, 
                    num_features=num_feats
                )

                label_to_use = target_class_idx
                if target_class_idx not in exp_p.local_exp:
                    if not exp_p.local_exp: return None
                    label_to_use = list(exp_p.local_exp.keys())[0]
                
                return dict(exp_p.as_list(label=label_to_use))
            except Exception:
                return None

        # Avoid nested parallelism; rely on outer level 
        perturbed_explanation_dicts = [
            get_single_lime_explanation(
                p.values.flatten() if hasattr(p, 'values') else np.array(p).flatten(),
                self.model.predict_proba,
                explainer,
                actual_class_idx,
                num_features_to_explain
            )
            for p in perturbations
        ]
        
        valid_perturbed_explanations = [d for d in perturbed_explanation_dicts if d is not None]

        if not original_weights or not valid_perturbed_explanations: 
            return {'mean_absolute_deviation': np.nan, 'cosine_similarity': np.nan, 'std_deviation': np.nan}

        return self._process_dict_explanations(original_weights, valid_perturbed_explanations)

    def batch_evaluate(self, X, explainer, method='shap', class_idx=None, n_jobs=-1):
        """
        Evaluate sensitivity for a batch of instances in parallel.
        method: 'shap' or 'lime'
        Returns: list of sensitivity dicts
        """
        if method == 'shap':
            func = lambda inst: self.evaluate_shap(inst, explainer, class_idx)
        elif method == 'lime':
            func = lambda inst: self.evaluate_lime(inst, explainer, class_idx)
        else:
            raise ValueError('Unknown method')
        results = Parallel(n_jobs=n_jobs)(delayed(func)(X.iloc[[i]]) for i in range(len(X)))
        return results

    def evaluate_dalex(self, instance, explainer):
        original_exp = explainer.predict_parts(instance, type='shap')
        perturbations = self._perturb_input(instance)

        perturbed_explanations = []
        for pert in perturbations:
            perturbed_explanations.append(explainer.predict_parts(pert, type='shap'))

        return self._calculate_sensitivity(
            original_exp.result['contribution'].values,
            [pe.result['contribution'].values for pe in perturbed_explanations]
        )

    def evaluate_ebm(self, instance, ebm_explainer):
        original_explanation = ebm_explainer.explain_local(instance).data
        perturbations = self._perturb_input(instance)

        perturbed_explanations = []
        for pert in perturbations:
            perturbed_explanations.append(ebm_explainer.explain_local(pert).data)

        return self._calculate_sensitivity(
            original_explanation['scores'],
            [pe['scores'] for pe in perturbed_explanations]
        )

    def _process_dict_explanations(self, original_dict, perturbed_dicts):
        all_features = set(original_dict.keys())
        for pd in perturbed_dicts:
            all_features.update(pd.keys())

        feature_vector = []
        for feat in all_features:
            feature_vector.append(original_dict.get(feat, 0))

        perturbed_vectors = []
        for pd in perturbed_dicts:
            pvec = [pd.get(feat, 0) for feat in all_features]
            perturbed_vectors.append(pvec)

        return self._calculate_sensitivity(
            np.array(feature_vector),
            np.array(perturbed_vectors)
        )

    def evaluate_all_classes(self, instance, explainer, method):
        """
        Evaluate sensitivity for all classes for a given instance.
        Returns: list of sensitivity dicts, one per class
        """
        n_classes = len(self.model.classes_)
        results = []
        for class_idx in range(n_classes):
            if method == 'shap':
                res = self.evaluate_shap(instance, explainer, class_idx)
            elif method == 'lime':
                res = self.evaluate_lime(instance, explainer, class_idx)
            else:
                raise ValueError('Diffrent/Unknown method')
            results.append(res)
        return results

