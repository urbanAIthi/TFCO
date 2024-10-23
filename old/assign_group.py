import random


names = list(range(1, 25))
# drop names 1,14,15s
print(len(names))
print(names)
random.shuffle(names)
groups = []
group = []

for name in names:
    group.append(name)
    # Create groups of 3 or 4. If group size is 3 or 4, start a new group.
    if len(group) == 3 or (len(group) == 4 and len(names) - len(group) * len(groups) > 2):
        groups.append(group)
        group = []

# Handle the remaining names if there are any
if group:
    # Distribute remaining members to already created groups if less than 3 members are left.
    for i, name in enumerate(group):
        groups[i].append(name)
for idx, group in enumerate(groups, start=1):
    print(f"Group {idx}: {', '.join(map(str, group))}")