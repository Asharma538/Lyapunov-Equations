# import matplotlib.pyplot as plt # type: ignore
# import matplotlib.ticker as ticker # type: ignore

# # Data
# n = [100, 200, 500, 1000, 2000, 5000]
# time_sequential = [980192, 6021692, 86042668, 700369813, 6206020827, 96969075421]
# time_parallel = [944886, 5378558, 82850872, 666736693, 5768161384, 90127521625]

# # Plotting
# plt.plot(n, time_sequential, label='Sequential Solver')
# plt.plot(n, time_parallel, label='Parallel Solver')

# # Labels and title
# plt.xlabel('n (size of matrix)')
# plt.ylabel('Time (microseconds)')
# plt.title('Execution Time by Sequential and Parallel Solvers')
# plt.legend()

# # Show plot
# plt.show()


n = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
size_matrices = [90000, 360000, 2250000, 9000000, 36000000, 225000000, 900000000, 3600000000, 22500000000]

plt.plot(n, size_matrices, marker='o')
for i, txt in enumerate(size_matrices):
    plt.annotate(f'{txt:.2e}', (n[i], size_matrices[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
plt.gca().yaxis.set_minor_formatter(ticker.ScalarFormatter(useMathText=True))
    
plt.gca().set_xscale('log')
plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
plt.gca().xaxis.set_minor_formatter(ticker.ScalarFormatter())

plt.xlabel('n (dimension of matrix)')
plt.ylabel('Size of matrix (bytes)')
plt.title('Space complexity Analysis')

plt.show()