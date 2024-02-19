BeginPackage["CommitteeIsserlis`"]
  LocalFieldsIntegral::usage = 
    "Compute the expectation over local fields of a function."
  IsserlisTheorem::usage = 
    "Compute the expectation of a polynomial over a multivariate gaussian."
  RawMoment::usage = 
    "Compute the moment of a multivariate Gaussian."
  IsserlisCovariance::usage = 
    "Compute the covariance of a polynomial over a multivariate gaussian."

  StudentHiddenUnits = 2; TeacherHiddenUnits = 1;

  (*The extra field is the artificial noise*)
  TotalDimension = StudentHiddenUnits+TeacherHiddenUnits+1;

  lfs = \[Lambda]
  lft = (Superscript[\[Lambda], t])


  (*Extended Local Fields*)
  LocalFields = Range[TotalDimension];
  For[ita = 1, ita <= StudentHiddenUnits, ita++, LocalFields[[ita]] = lfs[ita] ]; (* the iterator is not a local variable! Be careful!!*)
  For[itb = StudentHiddenUnits+1, itb <= StudentHiddenUnits+TeacherHiddenUnits, itb++, LocalFields[[itb]] = lft[itb-StudentHiddenUnits] ];
  LocalFields[[TotalDimension]] = \[Xi];

  (*Extended Covariance Matrix for local field and artificial noise*)
  Omega = Normal@SymmetrizedArray[{i_, j_} -> omega[i, j], {TotalDimension, TotalDimension}, Symmetric[{1, 2}] ] /. {
        omega[i_, j_] :> If[i <= StudentHiddenUnits,
          If[j <= StudentHiddenUnits,
            q[i,j],
            If[j == TotalDimension,
              0,
              m[i,j-StudentHiddenUnits]
            ]
          ],
          If[j == TotalDimension,
            If[i == TotalDimension, \[CapitalDelta], 0],
            \[Rho][i-StudentHiddenUnits,j-StudentHiddenUnits]
          ]
        ]
  };

  OmegaAssumptions = {
    (* Normalization is good *)
    Det[Omega] > 0, Det[Omega] \[Element] Reals,
    (* Positive variance *)
    \[CapitalDelta] > 0, \[CapitalDelta] \[Element] Reals,
    (* Real matrix *)
    Sequence@@Flatten[Table[\[Rho][i, j] \[Element] Reals, {i, TeacherHiddenUnits}, {j, TeacherHiddenUnits}] ],
    Sequence@@Flatten[Table[q[i, j] \[Element] Reals, {i, StudentHiddenUnits}, {j, StudentHiddenUnits}] ],
    Sequence@@Flatten[Table[m[i, j] \[Element] Reals, {i, StudentHiddenUnits}, {j, TeacherHiddenUnits}] ],
    (* Positive variance *)
    Sequence@@Table[\[Rho][i, i] > 0, {i, TeacherHiddenUnits}],
    Sequence@@Table[q[i, i] > 0, {i, StudentHiddenUnits}],
    (* Covariance Inequalities *)
    Sequence@@Flatten[Table[
      If[
        i < j,
        q[i, j]^2 < q[i,i]*q[j,j],
        True
      ], {i, StudentHiddenUnits}, {j, StudentHiddenUnits}
    ] ],
    Sequence@@Flatten[Table[
      If[
        i < j,
        \[Rho][i, j]^2 < \[Rho][i,i]*\[Rho][j,j],
        True
      ], {i, TeacherHiddenUnits}, {j, TeacherHiddenUnits}]
    ],
    Sequence@@Flatten[
      Table[m[i, j]^2 < q[i, i]*\[Rho][j, j], {i, StudentHiddenUnits}, {j, TeacherHiddenUnits}]
    ],
  }

  (* PDF of the locals fields *)
  LFpdf = 1/Sqrt[(2*Pi)^TotalDimension*Det[Omega] ] * Exp[-1/2*LocalFields.Inverse[Omega].LocalFields];
  
  (* Custom Transformations*)
  OppositeSquareRootRateoTrasform[expr_] := Module[{newExpr},
    newExpr = 
      expr /. {Sqrt[a_]/Sqrt[b_] :> 
        If[a == -b, I*Sign[b], Sqrt[a]/Sqrt[b] ]
      };
    Simplify[newExpr, TransformationFunctions -> {Automatic}] ];

  DeterminantSignTrasform[expr_] := Module[{newExpr, expandedDeterminant},
    expandedExpr = Expand[a];
    expandedDeterminant = Expand[Det[Omega]/\[CapitalDelta]];
    newExpr = 
      expr /. {Sign[a_] :> 
        If[
          expandedExpr == expandedDeterminant,
          1,
          If[expandedExpr == -expandedDeterminant, -1, a]
        ]
      };
    Simplify[newExpr, TransformationFunctions -> {Automatic}] ];

  
  (*The moment generating function for local fields*)
  LFMomentGeneratingFunction = Exp[1/2* LocalFields.Omega.LocalFields];
  
  Begin["Private`"]

  LocalFieldsIntegral[function_] :=
      (* FullSimplify[ *)Integrate[ 
          function*LFpdf,
          Sequence@@Table[{LocalFields[[i]], -Infinity, Infinity}, {i, TotalDimension}],
          Assumptions -> OmegaAssumptions
        ] /. {
          ArcCot[y_] :> ArcSin[Sqrt[1/(y^2 + 1)] ]
        }(*]*);
    
  End[]
EndPackage[]