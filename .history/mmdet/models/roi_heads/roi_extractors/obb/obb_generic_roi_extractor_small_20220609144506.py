import torch
from mmdet.core import force_fp32
from mmdet.models.builder import ROI_EXTRACTORS
from mmdet.ops.plugin import build_plugin_layer
from .obb_base_roi_extractor import OBBBaseRoIExtractor


@ROI_EXTRACTORS.register_module()
class OBBGenericRoIExtractorSmall(OBBBaseRoIExtractor):
    """Extract RoI features from all level feature maps levels.

    This is the implementation of `A novel Region of Interest Extraction Layer
    for Instance Segmentation <https://arxiv.org/abs/2004.13665>`_.

    Args:
        aggregation (str): The method to aggregate multiple feature maps.
            Options are 'sum', 'concat'. Default: 'sum'.
        pre_cfg (dict | None): Specify pre-processing modules. Default: None.
        post_cfg (dict | None): Specify post-processing modules. Default: None.
        kwargs (keyword arguments): Arguments that are the same
            as :class:`BaseRoIExtractor`.
    """

    def __init__(self,
                 aggregation='sum',
                 pre_cfg=None,
                 post_cfg=None,
                 roi_scale_factor = None,
                 **kwargs):
        super(OBBGenericRoIExtractorSmall, self).__init__(**kwargs)

        assert aggregation in ['sum', 'concat']

        self.roi_scale_factor = roi_scale_factor
        self.aggregation = aggregation
        self.with_post = post_cfg is not None
        self.with_pre = pre_cfg is not None
        # build pre/post processing modules
        if self.with_post:
            self.post_module = build_plugin_layer(post_cfg, '_post_module')[1]
        if self.with_pre:
            self.pre_module = build_plugin_layer(pre_cfg, '_pre_module')[1]

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois):

        if self.roi_scale_factor is not None:
            rois = self.roi_rescale(rois, self.roi_scale_factor)
            
        """Forward function"""
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)

        out_size = self.roi_layers[0].out_size
        num_levels = len(feats)
        
        rois_small = rois[rois[:,3] * rois[:,4] <= 6000]
        rois_big   = rois[rois[:,3] * rois[:,4] > 6000]
        
        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, *out_size)
        
        roi_feats_small = feats[0].new_zeros(
            rois_small.size(0), self.out_channels, *out_size)
        
        roi_feats_big = feats[0].new_zeros(
            rois_big.size(0), self.out_channels, *out_size)

        # some times rois is an empty tensor
        if roi_feats.shape[0] == 0:
            return roi_feats



        # mark the starting channels for concat mode
        start_channels = 0

        # for big rois
        for i in range(num_levels):
            roi_feats_t_big = self.roi_layers[i](feats[i], rois_big)
            end_channels = start_channels + roi_feats_t_big.size(1)
            if self.with_pre:
                # apply pre-processing to a RoI extracted from each layer
                roi_feats_t_big = self.pre_module(roi_feats_t_big)
            if self.aggregation == 'sum':
                # and sum them all
                roi_feats_big += roi_feats_t_big
            else:
                # and concat them along channel dimension
                roi_feats_big[:, start_channels:end_channels] = roi_feats_t_big
            # update channels starting position
            start_channels = end_channels
        # for small rois  
        for i in range(2):
            roi_feats_t_small = self.roi_layers[i](feats[i], rois_small)
            end_channels = start_channels + roi_feats_t_small.size(1)
            if self.with_pre:
                # apply pre-processing to a RoI extracted from each layer
                roi_feats_t_small = self.pre_module(roi_feats_t_small)
            if self.aggregation == 'sum':
                # and sum them all
                roi_feats_small += roi_feats_t_small
            else:
                # and concat them along channel dimension
                roi_feats_small[:, start_channels:end_channels] = roi_feats_t_small
            # update channels starting position
            start_channels = end_channels
           
        # check if concat channels match at the end
        if self.aggregation == 'concat':
            assert start_channels == self.out_channels
            
        roi_feats = torch.cat((roi_feats_big,roi_feats_small))
        
        if self.with_post:
            # apply post-processing before return the result
            roi_feats = self.post_module(roi_feats)
        return roi_feats
