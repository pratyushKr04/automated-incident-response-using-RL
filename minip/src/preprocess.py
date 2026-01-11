"""
Data Preprocessing Module for Automated Incident Response.

This module handles:
- Loading and processing CICIDS 2017 dataset for brute-force attack modeling
- Loading and processing CERT Insider Threat dataset for ransomware-like behavior
- Extracting behavioral features and fitting statistical distributions
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional
import json
import os
from pathlib import Path


class CICIDSPreprocessor:
    """
    Preprocessor for CICIDS 2017 dataset.
    
    Extracts brute-force login attack features:
    - Total Fwd Packets (proxy for login attempts)
    - Flow Duration (proxy for session persistence)
    """
    
    def __init__(self, monday_path: str, tuesday_path: str):
        """
        Initialize with paths to CICIDS CSV files.
        
        Args:
            monday_path: Path to Monday (benign) traffic file
            tuesday_path: Path to Tuesday (attack) traffic file
        """
        self.monday_path = monday_path
        self.tuesday_path = tuesday_path
        self.benign_data = None
        self.attack_data = None
        self.fitted_params = {}
    
    def load_data(self, sample_size: Optional[int] = 100000) -> None:
        """
        Load CICIDS datasets.
        
        Args:
            sample_size: Number of rows to sample (for memory efficiency)
        """
        print("Loading CICIDS 2017 datasets...")
        
        # Load Monday (benign traffic)
        try:
            monday_df = pd.read_csv(self.monday_path, low_memory=False)
            if sample_size and len(monday_df) > sample_size:
                monday_df = monday_df.sample(n=sample_size, random_state=42)
            self.benign_data = monday_df
            print(f"  Monday data loaded: {len(monday_df)} rows")
        except Exception as e:
            print(f"  Warning: Could not load Monday data: {e}")
            self.benign_data = None
        
        # Load Tuesday (brute-force attacks)
        try:
            tuesday_df = pd.read_csv(self.tuesday_path, low_memory=False)
            if sample_size and len(tuesday_df) > sample_size:
                tuesday_df = tuesday_df.sample(n=sample_size, random_state=42)
            self.attack_data = tuesday_df
            print(f"  Tuesday data loaded: {len(tuesday_df)} rows")
        except Exception as e:
            print(f"  Warning: Could not load Tuesday data: {e}")
            self.attack_data = None
    
    def extract_features(self) -> Dict[str, np.ndarray]:
        """
        Extract relevant features for brute-force modeling.
        
        Returns:
            Dictionary with extracted feature arrays
        """
        features = {}
        
        # Column name variations in CICIDS dataset
        fwd_packets_cols = [' Total Fwd Packets', 'Total Fwd Packets', 'total_fwd_packets']
        flow_duration_cols = [' Flow Duration', 'Flow Duration', 'flow_duration']
        label_cols = [' Label', 'Label', 'label']
        
        def get_column(df, candidates):
            for col in candidates:
                if col in df.columns:
                    return col
            return None
        
        if self.benign_data is not None:
            fwd_col = get_column(self.benign_data, fwd_packets_cols)
            dur_col = get_column(self.benign_data, flow_duration_cols)
            
            if fwd_col:
                features['benign_fwd_packets'] = pd.to_numeric(
                    self.benign_data[fwd_col], errors='coerce'
                ).dropna().values
            
            if dur_col:
                features['benign_flow_duration'] = pd.to_numeric(
                    self.benign_data[dur_col], errors='coerce'
                ).dropna().values
        
        if self.attack_data is not None:
            fwd_col = get_column(self.attack_data, fwd_packets_cols)
            dur_col = get_column(self.attack_data, flow_duration_cols)
            label_col = get_column(self.attack_data, label_cols)
            
            # Separate benign and attack flows from Tuesday
            if label_col:
                benign_mask = self.attack_data[label_col].str.strip().str.upper() == 'BENIGN'
                attack_mask = ~benign_mask
            else:
                # Assume all Tuesday data contains attacks
                attack_mask = np.ones(len(self.attack_data), dtype=bool)
            
            if fwd_col:
                attack_flows = self.attack_data[attack_mask]
                features['attack_fwd_packets'] = pd.to_numeric(
                    attack_flows[fwd_col], errors='coerce'
                ).dropna().values
            
            if dur_col:
                attack_flows = self.attack_data[attack_mask]
                features['attack_flow_duration'] = pd.to_numeric(
                    attack_flows[dur_col], errors='coerce'
                ).dropna().values
        
        return features
    
    def fit_poisson(self, data: np.ndarray) -> Tuple[float, float]:
        """
        Fit Poisson distribution to event count data.
        
        Args:
            data: Array of event counts
            
        Returns:
            Tuple of (lambda, goodness_of_fit_pvalue)
        """
        # Poisson lambda is estimated by sample mean
        data = data[data >= 0]  # Ensure non-negative
        data = data[data < np.percentile(data, 99)]  # Remove extreme outliers
        
        lambda_hat = np.mean(data)
        
        # Goodness of fit test (chi-square)
        try:
            observed_freq, bins = np.histogram(data, bins=20)
            expected_freq = len(data) * np.diff(stats.poisson.cdf(bins, lambda_hat))
            expected_freq = np.maximum(expected_freq, 1)  # Avoid division by zero
            chi2, p_value = stats.chisquare(observed_freq, expected_freq)
        except Exception:
            p_value = 0.0
        
        return lambda_hat, p_value
    
    def get_attack_parameters(self) -> Dict[str, Dict[str, float]]:
        """
        Extract parameters for attack simulation.
        
        Returns:
            Dictionary of fitted parameters
        """
        features = self.extract_features()
        params = {"bruteforce": {}}
        
        # Fit distributions for each feature
        if 'benign_fwd_packets' in features:
            lambda_benign, _ = self.fit_poisson(features['benign_fwd_packets'])
            params['bruteforce']['benign_login_rate'] = float(lambda_benign)
            print(f"  Benign login rate (λ): {lambda_benign:.2f}")
        
        if 'attack_fwd_packets' in features:
            lambda_attack, _ = self.fit_poisson(features['attack_fwd_packets'])
            params['bruteforce']['attack_login_rate'] = float(lambda_attack)
            print(f"  Attack login rate (λ): {lambda_attack:.2f}")
        
        self.fitted_params = params
        return params


class CERTPreprocessor:
    """
    Preprocessor for CERT Insider Threat dataset.
    
    Extracts ransomware-like behavioral features:
    - File access rates per time window
    - Duration of sustained activity bursts
    - Per-user activity patterns
    """
    
    def __init__(self, file_path: str):
        """
        Initialize with path to CERT file.csv.
        
        Args:
            file_path: Path to file.csv from CERT dataset
        """
        self.file_path = file_path
        self.data = None
        self.fitted_params = {}
    
    def load_data(self, sample_size: Optional[int] = 500000) -> None:
        """
        Load CERT file access log.
        
        Args:
            sample_size: Number of rows to sample
        """
        print("Loading CERT Insider Threat dataset...")
        
        try:
            # CERT file.csv typically has columns: id, date, user, pc, filename, activity
            df = pd.read_csv(self.file_path, low_memory=False)
            
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
            
            self.data = df
            print(f"  CERT data loaded: {len(df)} rows")
            print(f"  Columns: {list(df.columns)}")
        except Exception as e:
            print(f"  Warning: Could not load CERT data: {e}")
            self.data = None
    
    def extract_features(self, time_window_minutes: int = 1) -> Dict[str, np.ndarray]:
        """
        Extract file access rate features.
        
        Args:
            time_window_minutes: Size of time window for aggregation
            
        Returns:
            Dictionary with extracted features
        """
        features = {}
        
        if self.data is None:
            return features
        
        # Try to parse datetime
        date_cols = ['date', 'datetime', 'timestamp', 'Date']
        date_col = None
        for col in date_cols:
            if col in self.data.columns:
                date_col = col
                break
        
        if date_col:
            try:
                self.data['parsed_datetime'] = pd.to_datetime(self.data[date_col])
                
                # Create time window bins
                self.data['time_window'] = self.data['parsed_datetime'].dt.floor(
                    f'{time_window_minutes}min'
                )
                
                # Count file accesses per time window
                access_counts = self.data.groupby('time_window').size()
                features['file_access_rates'] = access_counts.values
                
                print(f"  Extracted {len(access_counts)} time windows")
                print(f"  Mean file access rate: {np.mean(features['file_access_rates']):.2f}")
                
            except Exception as e:
                print(f"  Warning: Could not parse datetime: {e}")
        
        # Per-user analysis
        user_cols = ['user', 'User', 'user_id']
        user_col = None
        for col in user_cols:
            if col in self.data.columns:
                user_col = col
                break
        
        if user_col:
            user_activity = self.data.groupby(user_col).size()
            features['user_activity_counts'] = user_activity.values
        
        return features
    
    def detect_burst_patterns(self) -> Dict[str, float]:
        """
        Detect sustained activity bursts that could indicate ransomware.
        
        Returns:
            Dictionary with burst statistics
        """
        features = self.extract_features(time_window_minutes=1)
        burst_stats = {}
        
        if 'file_access_rates' in features:
            rates = features['file_access_rates']
            
            # Normal vs burst threshold (using percentile)
            normal_threshold = np.percentile(rates, 75)
            burst_threshold = np.percentile(rates, 95)
            
            burst_stats['normal_rate'] = float(np.mean(rates[rates <= normal_threshold]))
            burst_stats['elevated_rate'] = float(np.mean(rates[
                (rates > normal_threshold) & (rates <= burst_threshold)
            ]))
            burst_stats['burst_rate'] = float(np.mean(rates[rates > burst_threshold]))
            
            print(f"  Normal rate: {burst_stats['normal_rate']:.2f}")
            print(f"  Elevated rate: {burst_stats['elevated_rate']:.2f}")
            print(f"  Burst rate: {burst_stats['burst_rate']:.2f}")
        
        return burst_stats
    
    def get_attack_parameters(self) -> Dict[str, Dict[str, float]]:
        """
        Extract parameters for ransomware simulation.
        
        Returns:
            Dictionary of fitted parameters
        """
        features = self.extract_features()
        burst_stats = self.detect_burst_patterns()
        
        params = {
            "ransomware": {
                "normal_file_rate": burst_stats.get('normal_rate', 5.0),
                "execution_file_rate": burst_stats.get('elevated_rate', 20.0),
                "encryption_file_rate": burst_stats.get('burst_rate', 100.0),
                "post_attack_rate": 10.0
            }
        }
        
        self.fitted_params = params
        return params


class DataPreprocessor:
    """
    Main preprocessor combining CICIDS and CERT data processing.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize with data directory.
        
        Args:
            data_dir: Directory containing the datasets (defaults to project_root/data)
        """
        # Use project root to find data directory (don't import get_config - circular dependency!)
        project_root = Path(__file__).parent.parent.resolve()
        
        if data_dir is None:
            data_dir = project_root / "data"
        else:
            data_dir = Path(data_dir)
        
        # Set up paths
        self.cicids_monday_path = str(data_dir / "Monday-WorkingHours.pcap_ISCX.csv")
        self.cicids_tuesday_path = str(data_dir / "Tuesday-WorkingHours.pcap_ISCX.csv")
        self.cert_file_path = str(data_dir / "file.csv")
        
        self.cicids_processor = CICIDSPreprocessor(
            self.cicids_monday_path,
            self.cicids_tuesday_path
        )
        self.cert_processor = CERTPreprocessor(self.cert_file_path)
        self.combined_params = {}
    
    def process_all(self, save_path: Optional[str] = None) -> Dict:
        """
        Process all datasets and extract parameters.
        
        Args:
            save_path: Optional path to save extracted parameters
            
        Returns:
            Combined parameters dictionary
        """
        print("\n" + "="*60)
        print("Starting Data Preprocessing Pipeline")
        print("="*60 + "\n")
        
        # Process CICIDS data
        print("[1/2] Processing CICIDS 2017 Dataset...")
        self.cicids_processor.load_data()
        bruteforce_params = self.cicids_processor.get_attack_parameters()
        
        print("\n[2/2] Processing CERT Insider Threat Dataset...")
        self.cert_processor.load_data()
        ransomware_params = self.cert_processor.get_attack_parameters()
        
        # Combine parameters
        # CPU values are modeled (not from datasets - CICIDS/CERT don't have CPU metrics)
        self.combined_params = {
            **bruteforce_params,
            **ransomware_params,
            "cpu_usage": {
                "normal_mean": 30.0,   # Synthetic: typical idle CPU
                "normal_std": 5.0,
                "attack_mean": 80.0,   # Synthetic: high CPU during attack (crypto, scanning)
                "attack_std": 5.0
            }
        }
        
        # Save parameters if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(self.combined_params, f, indent=2)
            print(f"\nParameters saved to: {save_path}")
        
        print("\n" + "="*60)
        print("Preprocessing Complete!")
        print("="*60)
        
        return self.combined_params


def preprocess_data(save_path: str = None) -> Dict:
    """
    Convenience function to run preprocessing pipeline.
    
    Args:
        save_path: Path to save extracted parameters (defaults to project_root/extracted_params.json)
        
    Returns:
        Extracted parameters dictionary
    """
    if save_path is None:
        # Default to project root
        project_root = Path(__file__).parent.parent.resolve()
        save_path = str(project_root / "extracted_params.json")
    
    preprocessor = DataPreprocessor()
    return preprocessor.process_all(save_path)


if __name__ == "__main__":
    # Run preprocessing when executed directly
    params = preprocess_data()
    print("\nExtracted Parameters:")
    print(json.dumps(params, indent=2))
