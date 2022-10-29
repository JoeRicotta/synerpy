"""
The synergy index is canonically associated with the uncontrolled
manifold analysis, but since it is a metric, it lives here.
"""

def synergy_index(vucm, vort):

    # normalizing per dimension
    vort = sum(vort / self.dim_ort)
    vucm = sum(vucm / self.dim_ucm)

    # delta v
    dv = (vucm - vort)/(vucm + vort)

    return dv
