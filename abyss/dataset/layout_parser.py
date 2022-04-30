class LayoutParser:
    """Create corresponding layout pattern"""

    def __init__(self, layout_definition: str):
        layout_definition = layout_definition.replace(' ', '')
        layout_definition = layout_definition.split('->')
        self.layout_definition = layout_definition

    def __call__(self):
        self._check_for_valid_definition()

    def _check_for_valid_definition(self):
        """Check if used words are valid"""
        allowed_names = ['case_folder', 'time_step', 'modality_folder', 'image_files', 'dicom_files']
        count_allowed_names = len(allowed_names)
        count_layout = len(self.layout_definition)
        diff = len(set(allowed_names) - set(self.layout_definition))
        if count_allowed_names != count_layout + diff:
            raise ValueError(f'Invalid name in meta->folder_layout, options: {allowed_names}')
