from typing import Dict


class Container__:
    pass


Container__.name_dict = {
    'NF2D_RandomA': '7 random Gaussians',
    'NF2D_RandomB': '8 random Gaussians',
    'NF2D_1Bumps': 'Simple Gaussian',
    'NF2D_2Bumps': '2 Gaussians',
    'NF2D_1Rect': 'Simple Uniform',
    'NF2D_3Rect': '3 Uniforms',
    'NF2D_4Connected1':'4 connected Gaussians',
    'NF2D_4Rect': '4 Uniforms',
    'NF2D_10Bumps': '10 Gaussians',
    'NF2D_Diagonal4': '4 Gaussians, diag',
    'NF2D_Row3': '3 Gaussians, row',
    'NF2D_Row4': '4 Gaussians, row',
    'Dim10aCenteredMVG': 'Simple Gaussian',
    'Dim10bLargeGaps': '7 Gaussians, far',
    'Dim10cSmallGaps': '7 Gaussians, close'
}


class Constants:
    @staticmethod
    def get_name_dict() -> Dict[str, str]:
        return Container__.name_dict
