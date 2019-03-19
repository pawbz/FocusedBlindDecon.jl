

"""
Update the toeplitz matrix A using vector x
"""
function toeplitz!(A, x)
   n = length(x)
   for i in 1:n
	   for j in i:n
	       A[i,j] = x[j-i+1]
	       A[j,i] = x[j-i+1]
       end
   end
   return A
end

