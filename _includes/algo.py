

def ms(a):
	if not a:
		return []
	l = len(a)
	if l == 1:
		return a
	mid = l/2
	a1 = ms(a[0:mid])
	a2 = ms(a[mid:l])
	return merge(a1, a2)


def merge(a1, a2):
	m = len(a1)
	n = len(a2)
	i = 0
	j = 0
	a = []
	while (i < m) and (j < n):
		if a1[i] <= a2[j]:
			a.append(a1[i])
			i = i + 1
		else:
			a.append(a2[j])
			j = j + 1
	for ii in range(i, m):
		a.append(a1[ii])
	for jj in range(j, n):
		a.append(a2[jj])
	return a


