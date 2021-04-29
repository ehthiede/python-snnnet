# git filter-branch --prune-empty -d /dev/shm/scratch \
#       --index-filter "git rm --cached -f --ignore-unmatch examples/qm9/data/qm9_chem_molecules_train_qm9.json" \
#       --tag-name-filter cat -- --all

git filter-branch -f --prune-empty -d /dev/shm/scratch \
      --index-filter "git rm --cached -f --ignore-unmatch examples/qm9/data/molecules_train_qm9_chemical_features.json" \
      --tag-name-filter cat -- --all
