
items = [[('#e6e6e6', 'name'),
          ('#e6e6e6', 'bounds'),
          ('#e6e6e6', 'n'),
          ('#e6e6e6', 'success.master'),
          ('#e6e6e6', 'success.ipopt'),
          ('#e6e6e6', '.'), ('#e6e6e6', 'f.master'),
          ('#e6e6e6', 'f.ipopt'),
          ('#e6e6e6', '.'),
          ('#e6e6e6', 'iter.master'),
          ('#e6e6e6', 'iter.ipopt'),
          ('#e6e6e6', '.'),
          ('#e6e6e6', 'msg.master'),
          ('#e6e6e6', 'msg.ipopt'),
          ('#e6e6e6', '.')],
         [('#E3F2FD', '3PK'),
          ('#E3F2FD', 'True'),
          ('#E3F2FD', '30'),
          ('#E3F2FD', 'True'),
          ('#E3F2FD', 'True'),
          ('#e6e6e6', ''),
          ('#00e600', '1.72011857E+00'),
          ('#80ff80', '1.72011900E+00'),
          ('#e6e6e6', ''),
          ('#00e600', 1),
          ('#ffb84d', 11),
          ('#e6e6e6', ''),
          ('#E3F2FD', 'Optimal Solution Found.'),
          ('#E3F2FD', 'Optimal Solution Found.'),
          ('#e6e6e6', '')]]

items = [[('#e6e6e6', 'name'),
          ('#e6e6e6', 'bounds'),
          ('#e6e6e6', 'n'),
          ('#e6e6e6', 'success.master'),
          ('#e6e6e6', 'success.ipopt'),
          ('#e6e6e6', '.'), ('#e6e6e6', 'f.master'),
          ('#e6e6e6', 'f.ipopt'),
          ('#e6e6e6', '.'),
          ('#e6e6e6', 'iter.master'),
          ('#e6e6e6', 'iter.ipopt'),
          ('#e6e6e6', '.'),
          ('#e6e6e6', 'msg.master'),
          ('#e6e6e6', 'msg.ipopt'),
          ('#e6e6e6', '.')],
         [('#E3F2FD', '3PK'),
          ('#E3F2FD', 'True'),
          ('#E3F2FD', '30'),
          ('#E3F2FD', 'True'),
          ('#E3F2FD', 'True'),
          ('#e6e6e6', ''),
          ('#00e600', '1.72011857E+00'),
          ('#80ff80', '1.72011900E+00'),
          ('#e6e6e6', ''),
          ('#00e600', 1),
          ('#ffb84d', 11),
          ('#e6e6e6', ''),
          ('#E3F2FD', 'Optimal Solution Found.'),
          ('#E3F2FD', 'Optimal Solution Found.'),
          ('#e6e6e6', '')]]

items = [[('header', 'name'), ('header', 'bounds'), ('header', 'n'), ('header', 'success.master'), ('header', 'success.ipopt'), ('header', '.'), ('header', 'f.master'), ('header', 'f.ipopt'), ('header', '.'), ('header', 'iter.master'), ('header', 'iter.ipopt'), ('header', '.'), ('header', 'msg.master'), ('header', 'msg.ipopt'), ('header', '.')], [('default1', 'AIRCRFTB'), ('default1', 'True'), ('default1', '8'), ('default1', 'True'), ('default1', 'True'), ('header', ''), ('ok', '1.89069318E-08'), ('best', '4.78824700E-25'), ('header', ''), ('poor', 58), ('best', 15), ('header', ''), ('default1', 'Optimal Solution Found.'), ('default1', 'Optimal Solution Found.'), ('header', '')], [('default2', 'AKIVA'), ('default2', 'False'), ('default2', '2'), ('default2', 'True'), ('default2', 'True'), ('header', ''), ('ok', '6.16604221E+00'), ('best', '6.16604200E+00'), ('header', ''), ('best', 6), ('best', 6), ('header', ''), ('default2', 'Optimal Solution Found.'), ('default2', 'Optimal Solution Found.'), ('header', '')]]

ncols = len(items[0])
color_row = "\\cellcolor{{{}}} {}"
color_row += "& \\cellcolor{{{}}} {}" * (ncols - 1) + "\\\\"

latex = "\\begin{tabular}{" + ncols * "c" + "}"
for ts in items:
    latex += color_row.format(*list(sum(ts, ())))
latex += "\\end{tabular}"