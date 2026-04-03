"""
Fairness Evaluation Framework for XAI Methods

Fairness measures whether XAI explanations are BIASED toward sensitive features
like race, gender, or age. Fair explanations shouldn't change dramatically when
only the protected attribute changes.


Implement fairness assessment tool to measure bias and discrimination
in XAI explanations (SHAP, LIME) against sensitive/protected features.

Key Features:
- Fairness evaluation via feature-flipping counterfactual analysis
- MAD and cosine similarity metrics for bias detection
- Multi-feature fairness comparison across protected attributes

Metrics Provided:
1. Mean Absolute Deviation (MAD): Explanation change magnitude
2. Cosine Similarity: Explanation alignment consistency
3. Combined Fairness Score: Normalized composite metric


Mathematical Foundation:
- Counterfactual analysis: Compare explanations when sensitive feature is flipped
- Lower MAD = more fair (less sensitive to protected attributes)
- Higher cosine similarity = more fair (consistent explanations)
- Fairness Score = (normalized_inverted_MAD + normalized_cosine) / 2



"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from scipy.spatial.distance import cosine

class XAIFairnessEvaluator:
    """
    Evaluates fairness of XAI methods (SHAP, LIME) against sensitive/irrelevant features.
    """
    
    def __init__(self, model, feature_names, sensitive_feature):
        self.model = model
        self.feature_names = list(feature_names)
        if sensitive_feature not in self.feature_names:
            raise ValueError(f"Sensitive feature '{sensitive_feature}' not in features list")
        self.sensitive_idx = self.feature_names.index(sensitive_feature)
        self._training_data = None
        self._training_mean = None
        self._training_median = None

    def _predict_proba_with_names(self, x):
        if isinstance(x, np.ndarray):
            return self.model.predict_proba(pd.DataFrame(x, columns=self.feature_names))
        elif isinstance(x, pd.DataFrame):
            return self.model.predict_proba(x)
        else:
            return self.model.predict_proba(pd.DataFrame(x, columns=self.feature_names))

    def _flip_feature(self, x):
        if self._training_data is None:
            raise ValueError("Training data statistics not initialized.")
        
        x2 = x.copy()
        col = self.sensitive_idx
        feature_values = self._training_data[:, col]
        
        if set(np.unique(feature_values)) <= {0, 1}:
            x2[col] = 1 - int(x[col])
        elif np.issubdtype(feature_values.dtype, np.number):
            x2[col] = np.median(feature_values)
        else:
            raise ValueError(f"Categorical sensitive features not supported.")
        return x2

    def evaluate_shap_fairness(self, explainer, instances, num_instances=100, class_idx=None):
        if instances.shape[0] < num_instances:
            raise ValueError(f"Not enough instances: requested {num_instances}, but only {instances.shape[0]} available")
        
        inst = instances.iloc[:num_instances].values
        mad_scores, cos_sims = [], []
        
        if self._training_data is None:
            self._training_data = instances.values
            self._training_mean = instances.mean().values
            self._training_median = instances.median().values
        
        for x in tqdm(inst, desc="SHAP Fairness"):
            try:
                x_orig = x.reshape(1, -1)
                x_flip = self._flip_feature(x).reshape(1, -1)
                
                raw = explainer.shap_values(np.vstack([x_orig, x_flip]))
                
                if isinstance(raw, list):
                    ci = class_idx if class_idx is not None else int(self.model.predict(x_orig)[0])
                    sv = raw[ci]
                    orig_sv, flip_sv = sv[0], sv[1]
                elif isinstance(raw, np.ndarray) and raw.ndim == 3:
                    ci = class_idx if class_idx is not None else int(self.model.predict(x_orig)[0])
                    orig_sv, flip_sv = raw[0, :, ci], raw[1, :, ci]
                elif isinstance(raw, np.ndarray) and raw.ndim == 2:
                    orig_sv, flip_sv = raw[0], raw[1]
                else:
                    raise ValueError(f"Unexpected SHAP output shape")
                
                mad = mean_absolute_error(orig_sv, flip_sv)
                mad_scores.append(mad)
                
                if np.allclose(orig_sv, 0) and np.allclose(flip_sv, 0):
                    cos_sims.append(1.0)
                elif np.allclose(orig_sv, 0) or np.allclose(flip_sv, 0):
                    cos_sims.append(0.0)
                else:
                    cos_sims.append(1 - cosine(orig_sv, flip_sv))
            except Exception as e:
                raise RuntimeError(f"Error processing instance: {str(e)}")
        
        return {
            'mad_scores': mad_scores,
            'cosine_sims': cos_sims,
            'avg_mad': np.mean(mad_scores),
            'std_mad': np.std(mad_scores),
            'avg_cosine': np.mean(cos_sims),
            'std_cosine': np.std(cos_sims),
            'num_instances': len(mad_scores)
        }

    def evaluate_lime_fairness(self, explainer, instances, num_instances=100, class_idx=None):
        if instances.shape[0] < num_instances:
            raise ValueError(f"Not enough instances: requested {num_instances}, but only {instances.shape[0]} available")
        
        inst = instances.iloc[:num_instances].values
        mad_scores, cos_sims = [], []
        
        if self._training_data is None:
            self._training_data = instances.values
            self._training_mean = instances.mean().values
            self._training_median = instances.median().values
        
        for x in tqdm(inst, desc="LIME Fairness"):
            try:
                x_orig = x
                x_flip = self._flip_feature(x)
                
                ci = class_idx if class_idx is not None else int(self.model.predict(x_orig.reshape(1, -1))[0])
                
                exp_o = explainer.explain_instance(
                    x_orig,
                    self._predict_proba_with_names,
                    labels=[ci],
                    num_features=len(self.feature_names),
                    num_samples=1000
                )
                exp_f = explainer.explain_instance(
                    x_flip,
                    self._predict_proba_with_names,
                    labels=[ci],
                    num_features=len(self.feature_names),
                    num_samples=1000
                )
                
                orig_map = dict(exp_o.as_list(label=ci))
                flip_map = dict(exp_f.as_list(label=ci))
                
                def extract_feature_value(map_dict, feature):
                    for k, v in map_dict.items():
                        if k == feature or k.split()[0] == feature or k.startswith(feature):
                            return v
                    return 0.0
                
                o_vec = np.array([extract_feature_value(orig_map, f) for f in self.feature_names])
                f_vec = np.array([extract_feature_value(flip_map, f) for f in self.feature_names])
                
                mad = mean_absolute_error(o_vec, f_vec)
                mad_scores.append(mad)
                
                if np.allclose(o_vec, 0) and np.allclose(f_vec, 0):
                    cos_sims.append(1.0)
                elif np.allclose(o_vec, 0) or np.allclose(f_vec, 0):
                    cos_sims.append(0.0)
                else:
                    cos_sims.append(1 - cosine(o_vec, f_vec))
            except Exception as e:
                raise RuntimeError(f"Error processing instance: {str(e)}")
        
        return {
            'mad_scores': mad_scores,
            'cosine_sims': cos_sims,
            'avg_mad': np.mean(mad_scores),
            'std_mad': np.std(mad_scores),
            'avg_cosine': np.mean(cos_sims),
            'std_cosine': np.std(cos_sims),
            'num_instances': len(mad_scores)
        }
    
    def plot_results(self, results, title="", save_path=None):
        sns.set(style="whitegrid", font_scale=1.2)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        mad = results['mad_scores']
        axes[0].hist(mad, bins=20, color='skyblue', edgecolor='black', alpha=0.8)
        axes[0].axvline(np.mean(mad), color='red', linestyle='--', linewidth=2, 
                       label=f"Mean={np.mean(mad):.4f}")
        axes[0].axvline(np.median(mad), color='green', linestyle=':', linewidth=2,
                       label=f"Median={np.median(mad):.4f}")
        axes[0].set_title(f"MAD (Lower = Fairer)\nMean={results['avg_mad']:.4f}, Std={results['std_mad']:.4f}", 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel("Mean Absolute Deviation", fontsize=12)
        axes[0].set_ylabel("Frequency", fontsize=12)
        axes[0].legend(fontsize=11, loc='upper right')
        axes[0].grid(True, linestyle='--', alpha=0.6)

        cos = results['cosine_sims']
        axes[1].hist(cos, bins=20, color='salmon', edgecolor='black', alpha=0.8)
        axes[1].axvline(np.mean(cos), color='red', linestyle='--', linewidth=2,
                       label=f"Mean={np.mean(cos):.4f}")
        axes[1].axvline(np.median(cos), color='green', linestyle=':', linewidth=2,
                       label=f"Median={np.median(cos):.4f}")
        axes[1].set_title(f"Cosine Similarity (Higher = Fairer)\nMean={results['avg_cosine']:.4f}, Std={results['std_cosine']:.4f}", 
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel("Cosine Similarity", fontsize=12)
        axes[1].set_ylabel("Frequency", fontsize=12)
        axes[1].legend(fontsize=11, loc='upper left')
        axes[1].set_xlim([-0.1, 1.1])

        fig.suptitle(f"Fairness Metric Distribution: {title}", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            
            print(f"Figure saved to: {save_path}")
        plt.show()

    def plot_fairness_comparison(self, shap_results, lime_results, save_path=None):
        """
        Plot comprehensive fairness comparison between SHAP and LIME.
        """
        sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.6})
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        shap_mad = shap_results['avg_mad']
        lime_mad = lime_results['avg_mad']
        shap_mad_std = shap_results['std_mad']
        lime_mad_std = lime_results['std_mad']
        
        shap_cos = shap_results['avg_cosine']
        lime_cos = lime_results['avg_cosine']
        shap_cos_std = shap_results['std_cosine']
        lime_cos_std = lime_results['std_cosine']
        
        x_pos = np.array([0, 1])
        width = 0.4
        
        # LEFT: MAD SCORES
        axes[0].bar(x_pos[0], shap_mad, width, label='SHAP', color='#3498db', alpha=0.85, edgecolor='black', linewidth=2)
        axes[0].bar(x_pos[1], lime_mad, width, label='LIME', color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=2)
        
        axes[0].errorbar(x_pos[0], shap_mad, yerr=shap_mad_std, fmt='none', color='darkblue', capsize=10, capthick=2.5, linewidth=2.5)
        axes[0].errorbar(x_pos[1], lime_mad, yerr=lime_mad_std, fmt='none', color='darkred', capsize=10, capthick=2.5, linewidth=2.5)
        
        mad_max = max(shap_mad + shap_mad_std, lime_mad + lime_mad_std)
        axes[0].text(x_pos[0], shap_mad + shap_mad_std + (mad_max * 0.05), f'{shap_mad:.4f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color='darkblue')
        axes[0].text(x_pos[1], lime_mad + lime_mad_std + (mad_max * 0.05), f'{lime_mad:.4f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color='darkred')
        
        axes[0].set_ylabel('Mean Absolute Deviation', fontsize=13, fontweight='bold')
        axes[0].set_title('MAD Score Comparison\n(Lower = Fairer)', fontsize=14, fontweight='bold', pad=15)
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(['SHAP', 'LIME'], fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=12, loc='upper right', framealpha=0.95, edgecolor='black')
        axes[0].grid(True, linestyle='--', alpha=0.5, axis='y')
        axes[0].set_ylim(0, mad_max * 1.3)
        
        winner_mad = 'SHAP' if shap_mad < lime_mad else 'LIME'
        color_mad = 'lightgreen' if winner_mad == 'SHAP' else 'lightcoral'
        axes[0].text(0.5, 0.95, f'✓ {winner_mad} is Fairer', transform=axes[0].transAxes,
                    fontsize=12, fontweight='bold', ha='center', va='top',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor=color_mad, alpha=0.75, edgecolor='black', linewidth=1.5))
        
        # RIGHT: COSINE SIMILARITY
        axes[1].bar(x_pos[0], shap_cos, width, label='SHAP', color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=2)
        axes[1].bar(x_pos[1], lime_cos, width, label='LIME', color='#f39c12', alpha=0.85, edgecolor='black', linewidth=2)
        
        axes[1].errorbar(x_pos[0], shap_cos, yerr=shap_cos_std, fmt='none', color='darkgreen', capsize=10, capthick=2.5, linewidth=2.5)
        axes[1].errorbar(x_pos[1], lime_cos, yerr=lime_cos_std, fmt='none', color='darkorange', capsize=10, capthick=2.5, linewidth=2.5)
        
        cos_max = max(shap_cos + shap_cos_std, lime_cos + lime_cos_std)
        axes[1].text(x_pos[0], shap_cos + shap_cos_std + (cos_max * 0.05), f'{shap_cos:.4f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color='darkgreen')
        axes[1].text(x_pos[1], lime_cos + lime_cos_std + (cos_max * 0.05), f'{lime_cos:.4f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color='darkorange')
        
        axes[1].set_ylabel('Cosine Similarity', fontsize=13, fontweight='bold')
        axes[1].set_title('Cosine Similarity Comparison\n(Higher = Fairer)', fontsize=14, fontweight='bold', pad=15)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(['SHAP', 'LIME'], fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=12, loc='lower right', framealpha=0.95, edgecolor='black')
        axes[1].grid(True, linestyle='--', alpha=0.5, axis='y')
        axes[1].set_ylim(0, min(1.1, cos_max * 1.2))
        
        winner_cos = 'SHAP' if shap_cos > lime_cos else 'LIME'
        color_cos = 'lightgreen' if winner_cos == 'SHAP' else 'lightcoral'
        axes[1].text(0.5, 0.05, f'✓ {winner_cos} is Fairer', transform=axes[1].transAxes,
                    fontsize=12, fontweight='bold', ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor=color_cos, alpha=0.75, edgecolor='black', linewidth=1.5))
        
        fig.suptitle('XAI Fairness Evaluation: SHAP vs LIME\n' + f'Sensitive Feature: {self.feature_names[self.sensitive_idx]}',
                    fontsize=16, fontweight='bold', y=0.98)
        
        info_text = f"n={shap_results['num_instances']} instances"
        fig.text(0.99, 0.01, info_text, fontsize=10, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7, pad=0.5, edgecolor='black', linewidth=1))
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        
        if save_path:
            print(f" Figure saved to: {save_path}")
        plt.show()

    def plot_multiple_features_comparison(self, results_dict, metric='mad', save_path=None):
        """
        Plot fairness comparison for MULTIPLE sensitive features.
        """
        if metric not in ['mad', 'cosine']:
            raise ValueError("metric must be 'mad' or 'cosine'")
        
        sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.6})
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        
        features = list(results_dict.keys())
        n_features = len(features)
        
        if metric == 'mad':
            shap_vals = [results_dict[f]['shap']['avg_mad'] for f in features]
            lime_vals = [results_dict[f]['lime']['avg_mad'] for f in features]
            shap_errs = [results_dict[f]['shap']['std_mad'] for f in features]
            lime_errs = [results_dict[f]['lime']['std_mad'] for f in features]
            ylabel = 'Mean Absolute Deviation'
            title = 'Fairness Comparison: MAD Scores Across All Features\n(Lower = Fairer)'
            color_shap, color_lime = '#3498db', '#e74c3c'
        else:
            shap_vals = [results_dict[f]['shap']['avg_cosine'] for f in features]
            lime_vals = [results_dict[f]['lime']['avg_cosine'] for f in features]
            shap_errs = [results_dict[f]['shap']['std_cosine'] for f in features]
            lime_errs = [results_dict[f]['lime']['std_cosine'] for f in features]
            ylabel = 'Cosine Similarity'
            title = 'Fairness Comparison: Cosine Similarity Across All Features\n(Higher = Fairer)'
            color_shap, color_lime = '#2ecc71', '#f39c12'
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        x = np.arange(n_features)
        width = 0.35
        
        ax.bar(x - width/2, shap_vals, width, label='SHAP', color=color_shap, alpha=0.85, edgecolor='black', linewidth=1.5)
        ax.bar(x + width/2, lime_vals, width, label='LIME', color=color_lime, alpha=0.85, edgecolor='black', linewidth=1.5)
        
        ax.errorbar(x - width/2, shap_vals, yerr=shap_errs, fmt='none', color='black', capsize=6, capthick=2, linewidth=2)
        ax.errorbar(x + width/2, lime_vals, yerr=lime_errs, fmt='none', color='black', capsize=6, capthick=2, linewidth=2)
        
        ax.set_xlabel('Sensitive Features', fontsize=13, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(features, fontsize=12, fontweight='bold', ha='center')
        ax.legend(fontsize=12, loc='upper left', framealpha=0.95, edgecolor='black')
        ax.grid(True, linestyle='--', alpha=0.5, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            print(f" Figure saved to: {save_path}")
        plt.show()

      

    def plot_multiple_features_comparison(self, results_dict, metric='mad', save_path=None):
        """
        Plot fairness comparison for MULTIPLE sensitive features.
        """
        if metric not in ['mad', 'cosine']:
            raise ValueError("metric must be 'mad' or 'cosine'")
        
        sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.6})
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        
        features = list(results_dict.keys())
        n_features = len(features)
        
        if metric == 'mad':
            shap_vals = [results_dict[f]['shap']['avg_mad'] for f in features]
            lime_vals = [results_dict[f]['lime']['avg_mad'] for f in features]
            shap_errs = [results_dict[f]['shap']['std_mad'] for f in features]
            lime_errs = [results_dict[f]['lime']['std_mad'] for f in features]
            ylabel = 'Mean Absolute Deviation'
            title = 'Fairness Comparison: MAD Scores Across All Features\n(Lower = Fairer)'
            color_shap, color_lime = '#3498db', '#e74c3c'
        else:
            shap_vals = [results_dict[f]['shap']['avg_cosine'] for f in features]
            lime_vals = [results_dict[f]['lime']['avg_cosine'] for f in features]
            shap_errs = [results_dict[f]['shap']['std_cosine'] for f in features]
            lime_errs = [results_dict[f]['lime']['std_cosine'] for f in features]
            ylabel = 'Cosine Similarity'
            title = 'Fairness Comparison: Cosine Similarity Across All Features\n(Higher = Fairer)'
            color_shap, color_lime = '#2ecc71', '#f39c12'
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        x = np.arange(n_features)
        width = 0.35
        
        ax.bar(x - width/2, shap_vals, width, label='SHAP', color=color_shap, alpha=0.85, edgecolor='black', linewidth=1.5)
        ax.bar(x + width/2, lime_vals, width, label='LIME', color=color_lime, alpha=0.85, edgecolor='black', linewidth=1.5)
        
        ax.errorbar(x - width/2, shap_vals, yerr=shap_errs, fmt='none', color='black', capsize=6, capthick=2, linewidth=2)
        ax.errorbar(x + width/2, lime_vals, yerr=lime_errs, fmt='none', color='black', capsize=6, capthick=2, linewidth=2)
        
        ax.set_xlabel('Sensitive Features', fontsize=13, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(features, fontsize=12, fontweight='bold', ha='center')
        ax.legend(fontsize=12, loc='upper left', framealpha=0.95, edgecolor='black')
        ax.grid(True, linestyle='--', alpha=0.5, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            print(f" Figure saved to: {save_path}")
        plt.show()

    def plot_comprehensive_fairness_analysis(self, results_dict, save_path=None):
        """
        Creates a comprehensive 2-plot visualization:
        Plot 1: Combined MAD + Cosine Similarity for all features
        Plot 2: Overall Fairness Score comparison
        """
        from sklearn.preprocessing import MinMaxScaler
        
        sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.6})
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        
        features = list(results_dict.keys())
        
        shap_mad = [results_dict[f]['shap']['avg_mad'] for f in features]
        lime_mad = [results_dict[f]['lime']['avg_mad'] for f in features]
        shap_cosine = [results_dict[f]['shap']['avg_cosine'] for f in features]
        lime_cosine = [results_dict[f]['lime']['avg_cosine'] for f in features]
        

        fig1, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        x = np.arange(len(features))
        width = 0.35
        
        # MAD 
        axes[0].bar(x - width/2, shap_mad, width, label='SHAP', color='#3498db', alpha=0.85, edgecolor='black', linewidth=1.5)
        axes[0].bar(x + width/2, lime_mad, width, label='LIME', color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.5)
        axes[0].set_xlabel('Sensitive Features', fontsize=13, fontweight='bold')
        axes[0].set_ylabel('Mean Absolute Deviation (MAD)', fontsize=13, fontweight='bold')
        axes[0].set_title('MAD Scores Across Features\n(Lower = Fairer)', fontsize=14, fontweight='bold', pad=15)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(features, rotation=45, ha='right', fontsize=11, fontweight='bold')
        axes[0].legend(fontsize=12, loc='upper left', framealpha=0.95, edgecolor='black')
        axes[0].grid(axis='y', alpha=0.4, linestyle='--')
        
        # Cosine Similarity 
        axes[1].bar(x - width/2, shap_cosine, width, label='SHAP', color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=1.5)
        axes[1].bar(x + width/2, lime_cosine, width, label='LIME', color='#f39c12', alpha=0.85, edgecolor='black', linewidth=1.5)
        axes[1].set_xlabel('Sensitive Features', fontsize=13, fontweight='bold')
        axes[1].set_ylabel('Cosine Similarity', fontsize=13, fontweight='bold')
        axes[1].set_title('Cosine Similarity Scores Across Features\n(Higher = Fairer)', fontsize=14, fontweight='bold', pad=15)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(features, rotation=45, ha='right', fontsize=11, fontweight='bold')
        axes[1].legend(fontsize=12, loc='lower left', framealpha=0.95, edgecolor='black')
        axes[1].grid(axis='y', alpha=0.4, linestyle='--')
        axes[1].set_ylim(0, 1.1)
        
        fig1.suptitle('Fairness Metrics Comparison: MAD and Cosine Similarity\nAcross All Sensitive Features',
                     fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_path:
            path1 = save_path.replace('.png', '_metrics.png')
            print(f" Metrics comparison saved to: {path1}")
        plt.show()
        

        scaler = MinMaxScaler()
        
        # Normalize scores (invert MAD: lower is better)
        shap_mad_norm = 1 - scaler.fit_transform(np.array(shap_mad).reshape(-1, 1)).flatten()
        lime_mad_norm = 1 - scaler.fit_transform(np.array(lime_mad).reshape(-1, 1)).flatten()
        shap_cosine_norm = scaler.fit_transform(np.array(shap_cosine).reshape(-1, 1)).flatten()
        lime_cosine_norm = scaler.fit_transform(np.array(lime_cosine).reshape(-1, 1)).flatten()
        

        shap_fairness_score = (shap_mad_norm + shap_cosine_norm) / 2
        lime_fairness_score = (lime_mad_norm + lime_cosine_norm) / 2
        
        fig2, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(features))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, shap_fairness_score, width, label='SHAP', 
                      color='#27ae60', alpha=0.85, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, lime_fairness_score, width, label='LIME', 
                      color='#c0392b', alpha=0.85, edgecolor='black', linewidth=1.5)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Sensitive Features', fontsize=13, fontweight='bold')
        ax.set_ylabel('Overall Fairness Score (0-1, Higher = Fairer)', fontsize=13, fontweight='bold')
        ax.set_title('Overall Fairness Comparison Across Features\n(Combined Normalized MAD + Cosine Similarity)', 
                    fontsize=15, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45, ha='right', fontsize=11, fontweight='bold')
        ax.legend(fontsize=12, loc='upper right', framealpha=0.95, edgecolor='black')
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.4, linestyle='--')
        
        # Add winner annotations
        for i in range(len(features)):
            if shap_fairness_score[i] > lime_fairness_score[i]:
                winner_x = x[i] - width/2
                winner_y = shap_fairness_score[i]
                winner_color = '#27ae60'
            else:
                winner_x = x[i] + width/2
                winner_y = lime_fairness_score[i]
                winner_color = '#c0392b'
            
            ax.plot(winner_x, winner_y + 0.08, marker='*', markersize=15, 
                   color='gold', markeredgecolor='black', markeredgewidth=1)
        
        plt.tight_layout()
        
        if save_path:
            path2 = save_path.replace('.png', '_overall_fairness.png')
            print(f" Overall fairness comparison saved to: {path2}")
        plt.show()
        
        # Return summary statistics
        return {
            'features': features,
            'shap_fairness_scores': shap_fairness_score.tolist(),
            'lime_fairness_scores': lime_fairness_score.tolist(),
            'shap_wins': sum(shap_fairness_score > lime_fairness_score),
            'lime_wins': sum(lime_fairness_score > shap_fairness_score),
            'ties': sum(shap_fairness_score == lime_fairness_score)
        }


    