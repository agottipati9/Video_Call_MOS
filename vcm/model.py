import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from vcm.vmaf_and_qr_code_alignment import get_vmaf, identify_dropped_frames

class VcmNet(nn.Module):
    def __init__(
            self, 
            hidden_size=256, 
            num_layers=6,
            pad=100,
            features = [
            'frame_skips', 'frame_freeze', 'integer_motion', 'integer_adm2', 'integer_adm_scale0', 'integer_adm_scale1',
            'integer_adm_scale2', 'integer_adm_scale3', 'integer_vif_scale0', 'integer_vif_scale1', 'integer_vif_scale2', 'integer_vif_scale3', 
            ]
            ):
        super().__init__()
        self.pad = pad
        self.features = features
        self.in_feat = len(features)
        self.layer_norm = nn.BatchNorm1d(self.in_feat)
        self.lstm = nn.LSTM(self.in_feat, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.layer_norm(x.transpose(2,1)).transpose(2,1)
        x = F.pad(x, (0,0,self.pad,0), "reflect")
        x, _ = self.lstm(x)
        x = x[:,self.pad:,:]
        x = self.fc(x).squeeze(2)
        mos_frame = x
        mos = x.mean(1)
        return mos, mos_frame
    
def get_features(df_features, features):

    def _get_frame_freeze_feature(x_ref_frames):
        x_prev = x_ref_frames[0]
        x_freezes = np.zeros(len(x_ref_frames), dtype=int)
        for i, x_cur in enumerate(x_ref_frames[1:], start=1):
            if x_cur == x_prev:
                x_freezes[i] = x_freezes[i-1] + 1
            x_prev = x_cur
        return x_freezes

    df_features['frame_skips'] = df_features['ref_frames'].diff().fillna(0)
    df_features['frame_freeze'] = _get_frame_freeze_feature(df_features['ref_frames'].to_numpy())
        
    x = df_features[features].to_numpy()
    return x, df_features

class VideoCallMosModel():
    def __init__(self, checkpoint=None):
        if checkpoint is None:
            try:
                import pkg_resources
                checkpoint = pkg_resources.resource_filename('vcm', 'video_call_mos_weights.pt')
            except pkg_resources.DistributionNotFound:
                checkpoint = 'vcm/video_call_mos_weights.pt'
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Running on {self.dev}")
        self.model = VcmNet()
        self.model.to(self.dev)
        self.model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))
        self.model.eval()
        print(f"LSTM model loaded from checkpoint {checkpoint}")

    def __call__(self, deg_video, ref_video, results_dir, tmp_dir, verbosity):
        os.makedirs(results_dir, exist_ok=True)
        if verbosity:
            print(f"Computing Video Call MOS for video {deg_video} with reference {ref_video}")

        # run reference alignment via QR-code detection and compute VMAF
        df_results = get_vmaf(deg_video, ref_video, tmp_dir, verbosity=verbosity)

        # Identify dropped frames
        complete_df = identify_dropped_frames(df_results)
        
        # Create a mask for non-dropped frames to compute their MOS normally
        non_dropped_mask = ~complete_df['is_dropped']
        df_for_mos = complete_df[non_dropped_mask].copy()
        
        # Run Video Call MOS LSTM based on VMAF features and alignment indices for non-dropped frames
        x, _ = get_features(df_for_mos, self.model.features)        

        # run Video Call MOS LSTM based on VMAF features and alignment indices
        # x = get_features(df_results, self.model.features)[0]
        x = torch.tensor(x, dtype=torch.float).unsqueeze(0).to(self.dev)
        with torch.no_grad():
            mos, mos_frame = self.model(x)
        mos = mos[0].cpu().numpy()
        mos_frame = mos_frame[0].cpu().numpy()

        # save results to CSV
        # Initialize video_call_mos column with 1.0 (for dropped frames)
        complete_df['video_call_mos'] = 1.0
        
        # Assign calculated MOS scores to non-dropped frames
        complete_df.loc[non_dropped_mask, 'video_call_mos'] = mos_frame
        # df_results['video_call_mos'] = mos_frame
        results_csv = os.path.join(results_dir, os.path.basename(deg_video).split('.')[0]+'.csv')
        # df_results.to_csv(results_csv, index=False)
        complete_df.to_csv(results_csv, index=False)
        complete_mos = complete_df['video_call_mos'].mean()

        if verbosity:
            print(f"Video Call MOS computation done. Results saved at {results_csv}")            
        return complete_mos, results_csv
