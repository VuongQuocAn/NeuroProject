from __future__ import annotations

from .xai_finer_cam import FinerCAMExplainer
from .xai_seg_eigen_cam import SegEigenCAMExplainer
from .xai_odam import ODAMExplainer


class XAIFactory:
    @staticmethod
    def classification(classifier) -> FinerCAMExplainer:
        return FinerCAMExplainer(classifier)

    @staticmethod
    def segmentation(segmentor) -> SegEigenCAMExplainer:
        return SegEigenCAMExplainer(segmentor)

    @staticmethod
    def detection(detector) -> ODAMExplainer:
        return ODAMExplainer(detector)
