import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def generate_complex_dataset():
    """
    Generate a complex 500-row network flow dataset that will challenge ML models.
    Features designed complexity:
    1. Severe class imbalance (15% malicious, 85% normal)
    2. Subtle feature correlations
    3. Realistic noise and outliers
    4. Multiple attack types with overlapping signatures
    5. Temporal attack patterns
    6. Edge cases and boundary conditions
    """
    
    np.random.seed(42)  # For reproducibility
    random.seed(42)
    
    rows = []
    
    # Define attack types and their characteristics
    attack_types = {
        'DDoS': {'flow_duration_range': (100000, 500000), 'pkt_count_multiplier': 5},
        'Port_Scan': {'flow_duration_range': (1000, 10000), 'pkt_count_multiplier': 0.1},
        'Data_Exfiltration': {'flow_duration_range': (1000000, 5000000), 'pkt_count_multiplier': 2},
        'Botnet': {'flow_duration_range': (50000, 200000), 'pkt_count_multiplier': 1.5},
        'Intrusion': {'flow_duration_range': (10000, 100000), 'pkt_count_multiplier': 0.8}
    }
    
    data_sources = ['PFCP_Establishment_DoS_15-sec_NRF', 'PFCP_Deletion_DoS_20-sec_UDR', 
                   'PFCP_Establishment_DoS_20-sec_NRF', 'PFCP_Establishment_DoS_15-sec_N4',
                   'HTTP_Flood', 'SYN_Flood', 'UDP_Flood', 'ICMP_Flood', 'Normal_Traffic']
    
    # IP pools for realistic network simulation
    src_ips = [f"172.21.0.{i}" for i in range(100, 130)]
    dst_ips = [f"172.21.0.{i}" for i in range(100, 130)] + [f"10.0.0.{i}" for i in range(1, 50)]
    
    for i in range(500):
        # Determine if this will be malicious (15% chance for severe imbalance)
        is_malicious = random.random() < 0.15
        
        if is_malicious:
            # Choose attack type
            attack_type = random.choice(list(attack_types.keys()))
            attack_chars = attack_types[attack_type]
            label = 'Malicious'
            data_source = random.choice(data_sources[:-1])  # Exclude normal traffic
            
            # Malicious flow characteristics
            flow_duration = random.randint(*attack_chars['flow_duration_range'])
            base_pkt_count = max(1, int(10 * attack_chars['pkt_count_multiplier']))
            
            # Add correlation patterns that are hard to detect
            if attack_type == 'DDoS':
                tot_fwd_pkts = random.randint(base_pkt_count * 50, base_pkt_count * 200)
                tot_bwd_pkts = random.randint(1, 5)  # Low response
            elif attack_type == 'Port_Scan':
                tot_fwd_pkts = random.randint(1, 3)
                tot_bwd_pkts = random.randint(0, 1)
            else:
                tot_fwd_pkts = random.randint(base_pkt_count, base_pkt_count * 10)
                tot_bwd_pkts = random.randint(base_pkt_count // 2, base_pkt_count * 2)
                
        else:
            # Normal traffic
            label = 'Normal'
            data_source = 'Normal_Traffic'
            flow_duration = random.randint(1000, 1000000)
            tot_fwd_pkts = random.randint(1, 50)
            tot_bwd_pkts = random.randint(1, 50)
        
        # Add realistic noise to make patterns harder to detect
        noise_factor = random.uniform(0.8, 1.2)
        tot_fwd_pkts = max(1, int(tot_fwd_pkts * noise_factor))
        tot_bwd_pkts = max(0, int(tot_bwd_pkts * noise_factor))
        
        # Generate correlated features with subtle patterns
        src_ip = random.choice(src_ips)
        dst_ip = random.choice(dst_ips)
        src_port = random.randint(1024, 65535)
        dst_port = random.choice([80, 443, 21, 22, 23, 25, 53, 110, 143, 993, 995] + 
                                list(range(1024, 65535))[:100])
        protocol = random.choice([6, 17, 1])  # TCP, UDP, ICMP
        
        # Packet length characteristics
        if is_malicious and attack_type == 'DDoS':
            fwd_pkt_len_max = random.randint(1400, 1500)  # MTU-sized packets
            fwd_pkt_len_min = fwd_pkt_len_max - random.randint(0, 100)
        else:
            fwd_pkt_len_max = random.randint(40, 1500)
            fwd_pkt_len_min = random.randint(40, min(fwd_pkt_len_max, 200))
            
        fwd_pkt_len_mean = (fwd_pkt_len_max + fwd_pkt_len_min) / 2 + random.uniform(-50, 50)
        fwd_pkt_len_std = abs(random.uniform(0, (fwd_pkt_len_max - fwd_pkt_len_min) / 2))
        
        # Backward packet characteristics
        bwd_pkt_len_max = random.randint(40, min(1500, fwd_pkt_len_max + 200))
        bwd_pkt_len_min = random.randint(40, min(bwd_pkt_len_max, 200))
        bwd_pkt_len_mean = (bwd_pkt_len_max + bwd_pkt_len_min) / 2 + random.uniform(-30, 30)
        bwd_pkt_len_std = abs(random.uniform(0, (bwd_pkt_len_max - bwd_pkt_len_min) / 2))
        
        # Calculate derived features
        total_pkts = tot_fwd_pkts + tot_bwd_pkts
        total_len = int(tot_fwd_pkts * fwd_pkt_len_mean + tot_bwd_pkts * bwd_pkt_len_mean)
        
        flow_byts_s = total_len / (flow_duration / 1000000) if flow_duration > 0 else 0
        flow_pkts_s = total_pkts / (flow_duration / 1000000) if flow_duration > 0 else 0
        
        # IAT (Inter-Arrival Time) features with realistic patterns
        if tot_fwd_pkts > 1:
            fwd_iat_mean = flow_duration / (tot_fwd_pkts - 1) if tot_fwd_pkts > 1 else 0
            fwd_iat_std = abs(random.uniform(0, fwd_iat_mean))
            fwd_iat_max = fwd_iat_mean + 2 * fwd_iat_std
            fwd_iat_min = max(0, fwd_iat_mean - fwd_iat_std)
        else:
            fwd_iat_mean = fwd_iat_std = fwd_iat_max = fwd_iat_min = 0
            
        if tot_bwd_pkts > 1:
            bwd_iat_mean = flow_duration / (tot_bwd_pkts - 1) if tot_bwd_pkts > 1 else 0
            bwd_iat_std = abs(random.uniform(0, bwd_iat_mean))
            bwd_iat_max = bwd_iat_mean + 2 * bwd_iat_std
            bwd_iat_min = max(0, bwd_iat_mean - bwd_iat_std)
        else:
            bwd_iat_mean = bwd_iat_std = bwd_iat_max = bwd_iat_min = 0
        
        # Flow IAT
        flow_iat_mean = (fwd_iat_mean + bwd_iat_mean) / 2 if (fwd_iat_mean + bwd_iat_mean) > 0 else 0
        flow_iat_std = abs(random.uniform(0, flow_iat_mean))
        flow_iat_max = flow_iat_mean + 2 * flow_iat_std
        flow_iat_min = max(0, flow_iat_mean - flow_iat_std)
        
        # Flag counts with attack-specific patterns
        if is_malicious:
            if attack_type == 'DDoS':
                syn_flag_cnt = random.randint(tot_fwd_pkts // 2, tot_fwd_pkts)
                ack_flag_cnt = random.randint(0, tot_bwd_pkts)
                rst_flag_cnt = random.randint(0, 2)
            elif attack_type == 'Port_Scan':
                syn_flag_cnt = tot_fwd_pkts
                ack_flag_cnt = 0
                rst_flag_cnt = tot_bwd_pkts
            else:
                syn_flag_cnt = random.randint(0, 2)
                ack_flag_cnt = random.randint(total_pkts // 2, total_pkts)
                rst_flag_cnt = random.randint(0, 1)
        else:
            syn_flag_cnt = random.randint(0, 2)
            ack_flag_cnt = random.randint(total_pkts // 2, total_pkts)
            rst_flag_cnt = random.randint(0, 1)
            
        fin_flag_cnt = random.randint(0, 2)
        psh_flag_cnt = random.randint(0, total_pkts // 3)
        urg_flag_cnt = random.randint(0, 1)
        cwe_flag_cnt = random.randint(0, 1)
        ece_flag_cnt = random.randint(0, 1)
        
        # Header lengths
        fwd_header_len = tot_fwd_pkts * random.randint(20, 60)
        bwd_header_len = tot_bwd_pkts * random.randint(20, 60)
        
        # Packet size statistics
        pkt_len_min = min(fwd_pkt_len_min, bwd_pkt_len_min) if tot_bwd_pkts > 0 else fwd_pkt_len_min
        pkt_len_max = max(fwd_pkt_len_max, bwd_pkt_len_max)
        pkt_len_mean = (fwd_pkt_len_mean * tot_fwd_pkts + bwd_pkt_len_mean * tot_bwd_pkts) / total_pkts if total_pkts > 0 else 0
        pkt_len_std = abs(random.uniform(0, pkt_len_mean / 2))
        pkt_len_var = pkt_len_std ** 2
        
        # Additional complex features
        down_up_ratio = tot_bwd_pkts / tot_fwd_pkts if tot_fwd_pkts > 0 else 0
        pkt_size_avg = total_len / total_pkts if total_pkts > 0 else 0
        fwd_seg_size_avg = (tot_fwd_pkts * fwd_pkt_len_mean) / tot_fwd_pkts if tot_fwd_pkts > 0 else 0
        bwd_seg_size_avg = (tot_bwd_pkts * bwd_pkt_len_mean) / tot_bwd_pkts if tot_bwd_pkts > 0 else 0
        
        # Subflow features
        subflow_fwd_pkts = tot_fwd_pkts
        subflow_fwd_byts = int(tot_fwd_pkts * fwd_pkt_len_mean)
        subflow_bwd_pkts = tot_bwd_pkts
        subflow_bwd_byts = int(tot_bwd_pkts * bwd_pkt_len_mean)
        
        # Window sizes
        init_fwd_win_byts = random.randint(1024, 65535) if tot_fwd_pkts > 0 else -1
        init_bwd_win_byts = random.randint(1024, 65535) if tot_bwd_pkts > 0 else -1
        
        # Active data packets
        fwd_act_data_pkts = max(0, tot_fwd_pkts - syn_flag_cnt - fin_flag_cnt - rst_flag_cnt)
        fwd_seg_size_min = fwd_pkt_len_min if tot_fwd_pkts > 0 else 0
        
        # Activity patterns (for temporal correlation)
        if is_malicious and attack_type in ['DDoS', 'Botnet']:
            active_mean = random.uniform(100000, 500000)
            active_std = random.uniform(0, active_mean / 3)
            idle_mean = random.uniform(0, 10000)
            idle_std = random.uniform(0, idle_mean)
        else:
            active_mean = random.uniform(1000, 100000)
            active_std = random.uniform(0, active_mean / 2)
            idle_mean = random.uniform(10000, 1000000)
            idle_std = random.uniform(0, idle_mean / 2)
            
        active_max = active_mean + 2 * active_std
        active_min = max(0, active_mean - active_std)
        idle_max = idle_mean + 2 * idle_std
        idle_min = max(0, idle_mean - idle_std)
        
        # Generate timestamp with realistic progression
        base_time = datetime(2022, 10, 4, 20, 0, 0)
        timestamp_offset = timedelta(seconds=i * random.randint(1, 300))
        timestamp = (base_time + timestamp_offset).strftime("%d-%m-%Y %H:%M")
        
        # Create flow ID
        flow_id = f"{src_ip}-{dst_ip}-{src_port}-{dst_port}-{protocol}"
        
        row = [
            i,  # Serial No
            flow_id,  # Flow ID
            src_ip,  # Src IP
            src_port,  # Src Port
            dst_ip,  # Dst IP
            dst_port,  # Dst Port
            protocol,  # Protocol
            timestamp,  # Timestamp
            flow_duration,  # Flow Duration
            tot_fwd_pkts,  # Tot Fwd Pkts
            tot_bwd_pkts,  # Tot Bwd Pkts
            subflow_fwd_byts,  # TotLen Fwd Pkts
            subflow_bwd_byts,  # TotLen Bwd Pkts
            fwd_pkt_len_max,  # Fwd Pkt Len Max
            fwd_pkt_len_min,  # Fwd Pkt Len Min
            round(fwd_pkt_len_mean, 2),  # Fwd Pkt Len Mean
            round(fwd_pkt_len_std, 2),  # Fwd Pkt Len Std
            bwd_pkt_len_max,  # Bwd Pkt Len Max
            bwd_pkt_len_min,  # Bwd Pkt Len Min
            round(bwd_pkt_len_mean, 2),  # Bwd Pkt Len Mean
            round(bwd_pkt_len_std, 2),  # Bwd Pkt Len Std
            round(flow_byts_s, 2),  # Flow Byts/s
            round(flow_pkts_s, 6),  # Flow Pkts/s
            round(flow_iat_mean, 2),  # Flow IAT Mean
            round(flow_iat_std, 2),  # Flow IAT Std
            round(flow_iat_max, 2),  # Flow IAT Max
            round(flow_iat_min, 2),  # Flow IAT Min
            int(flow_duration),  # Fwd IAT Tot
            round(fwd_iat_mean, 2),  # Fwd IAT Mean
            round(fwd_iat_std, 2),  # Fwd IAT Std
            round(fwd_iat_max, 2),  # Fwd IAT Max
            round(fwd_iat_min, 2),  # Fwd IAT Min
            int(flow_duration),  # Bwd IAT Tot
            round(bwd_iat_mean, 2),  # Bwd IAT Mean
            round(bwd_iat_std, 2),  # Bwd IAT Std
            round(bwd_iat_max, 2),  # Bwd IAT Max
            round(bwd_iat_min, 2),  # Bwd IAT Min
            random.randint(0, tot_fwd_pkts),  # Fwd PSH Flags
            random.randint(0, tot_bwd_pkts),  # Bwd PSH Flags
            random.randint(0, 1),  # Fwd URG Flags
            random.randint(0, 1),  # Bwd URG Flags
            fwd_header_len,  # Fwd Header Len
            bwd_header_len,  # Bwd Header Len
            round(tot_fwd_pkts / (flow_duration / 1000000), 6) if flow_duration > 0 else 0,  # Fwd Pkts/s
            round(tot_bwd_pkts / (flow_duration / 1000000), 6) if flow_duration > 0 else 0,  # Bwd Pkts/s
            pkt_len_min,  # Pkt Len Min
            pkt_len_max,  # Pkt Len Max
            round(pkt_len_mean, 2),  # Pkt Len Mean
            round(pkt_len_std, 2),  # Pkt Len Std
            round(pkt_len_var, 2),  # Pkt Len Var
            fin_flag_cnt,  # FIN Flag Cnt
            syn_flag_cnt,  # SYN Flag Cnt
            rst_flag_cnt,  # RST Flag Cnt
            psh_flag_cnt,  # PSH Flag Cnt
            ack_flag_cnt,  # ACK Flag Cnt
            urg_flag_cnt,  # URG Flag Cnt
            cwe_flag_cnt,  # CWE Flag Count
            ece_flag_cnt,  # ECE Flag Cnt
            round(down_up_ratio, 2),  # Down/Up Ratio
            round(pkt_size_avg, 2),  # Pkt Size Avg
            round(fwd_seg_size_avg, 2),  # Fwd Seg Size Avg
            round(bwd_seg_size_avg, 2),  # Bwd Seg Size Avg
            0,  # Fwd Byts/b Avg
            0,  # Fwd Pkts/b Avg
            0,  # Fwd Blk Rate Avg
            0,  # Bwd Byts/b Avg
            0,  # Bwd Pkts/b Avg
            0,  # Bwd Blk Rate Avg
            subflow_fwd_pkts,  # Subflow Fwd Pkts
            subflow_fwd_byts,  # Subflow Fwd Byts
            subflow_bwd_pkts,  # Subflow Bwd Pkts
            subflow_bwd_byts,  # Subflow Bwd Byts
            init_fwd_win_byts,  # Init Fwd Win Byts
            init_bwd_win_byts,  # Init Bwd Win Byts
            fwd_act_data_pkts,  # Fwd Act Data Pkts
            fwd_seg_size_min,  # Fwd Seg Size Min
            round(active_mean, 1),  # Active Mean
            round(active_std, 1),  # Active Std
            round(active_max, 1),  # Active Max
            round(active_min, 1),  # Active Min
            round(idle_mean, 1),  # Idle Mean
            round(idle_std, 1),  # Idle Std
            round(idle_max, 1),  # Idle Max
            round(idle_min, 1),  # Idle Min
            label,  # Label
            data_source  # Data_Source
        ]
        
        rows.append(row)
    
    # Create DataFrame
    columns = [
        'Serial No', 'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port',
        'Protocol', 'Timestamp', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
        'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min',
        'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min',
        'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s',
        'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
        'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
        'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
        'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
        'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s',
        'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var',
        'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt',
        'ACK Flag Cnt', 'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt',
        'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg',
        'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg',
        'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg',
        'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts',
        'Subflow Bwd Byts', 'Init Fwd Win Byts', 'Init Bwd Win Byts',
        'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean', 'Active Std',
        'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
        'Label', 'Data_Source'
    ]
    
    df = pd.DataFrame(rows, columns=columns)
    
    # Save to files
    raw_path = os.path.join("data", "raw", "ultra_complex_network_dataset.csv")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    df.to_csv(raw_path, index=False)
    
    # Also save in root for easy access
    root_path = "ultra_complex_network_dataset.csv"
    df.to_csv(root_path, index=False)
    
    print(f"✅ Complex 500-row dataset created successfully!")
    print(f"📁 Saved to: {raw_path}")
    print(f"📁 Also saved to: {root_path}")
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total rows: {len(df)}")
    print(f"   Normal samples: {len(df[df['Label'] == 'Normal'])}")
    print(f"   Malicious samples: {len(df[df['Label'] == 'Malicious'])}")
    print(f"   Class imbalance ratio: {len(df[df['Label'] == 'Malicious']) / len(df) * 100:.1f}% malicious")
    print(f"   Unique attack sources: {df[df['Label'] == 'Malicious']['Data_Source'].nunique()}")
    print(f"   Feature complexity: High correlation, noise, temporal patterns")
    print(f"\n🔥 This dataset will make your model sweat! Good luck! 💪")
    
    return df

if __name__ == "__main__":
    generate_complex_dataset()