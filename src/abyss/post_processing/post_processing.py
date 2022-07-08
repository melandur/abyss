import SimpleITK as sitk


def apply_largest_connected_component_filter(mask: sitk.Image, label: int = 1) -> sitk.Image:
    """Return largest connected component for single label"""
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)  # True is less restrictive, gives fewer connected components
    threshold = sitk.BinaryThreshold(mask, label, label, label, 0)
    lesions = cc_filter.Execute(threshold)
    rl_filter = sitk.RelabelComponentImageFilter()
    lesions = rl_filter.Execute(lesions)  # sort by size
    filtered_mask = sitk.BinaryThreshold(lesions, label, label, label, 0)
    return filtered_mask
