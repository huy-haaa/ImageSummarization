### submodular-BestOfBothWorlds

_submodular-BestOfBothWorlds_ là một thư viện python về triển khai song song MPI hiệu suất cao của thuật toán tiên tiến **LS+PGB** để tối đa hóa mô-đun phụ được mô tả trong bài báo **_"Best of Both Worlds: Tối đa hóa mô-đun phụ thực tế và tối ưu theo lý thuyết song song"_**.

### Prerequisites for replicating our Experiments:

Cài đặt các thư viện sau đây để triển khai code

-   Cài đặt **OpenMPI** bằng lệnh _apt install libopenmpi-dev_ (nếu bạn dùng Ubuntu)

-   Cài đặt **pandas**

-   Cài đặt **mpi4py**

-   Cài đặt **scipy**

-   Cài đặt **networkx**

-   Tải file **"images_10K_mat.csv"** ([download](https://drive.google.com/file/d/1L5EkO2XZc14malxAVJbtX9DaNYfT4ofK/view?usp=sharing)) và **"tweets_keyword_edgelist.txt"** ([download](https://drive.google.com/file/d/1fCLX_lQko87Ym1T_KgBEs5_0KUPHoutI/view?usp=sharing)) và chuyển vào thư mục _"submodular/data"_ .

### Thực hiện các thực nghiệm:

Triển khai bằng cách chạy các dòng lệnh sau

**Với thí nghiệm 1:** (Figure 2, 3, 4)

-   bash _./bash_scripts/run_ExpSet1.bash ExpSet1.py "**nThreads**"_
-   Thay thế **nThreads** bằng số luồng mà bạn muốn các thí nghiệm sử dụng

**To replicate Experiments set 2:** (Figure 5)

-   bash _./bash_scripts/run_ExpSet2.bash ExpSet2.py_
-   Tập lệnh yêu cầu tối đa 64 luồng để chạy. Nếu bạn muốn thay đổi điều đó, hãy thay đổi dòng **declare -a nT=(1 2 4 8 16 32 64)** trong _bash_scripts/run_ExpSet2.bash_

**Để hiển thị kết quả của bảng 2**
-Tạo kết quả cho **Thực nghiệm 1**

-   bash _bash_scripts/run_perf.bash_

Dữ liệu kết quả sẽ được tự động lưu dưới dạng tệp CSV trong thư mục **experiment_results_output_data** và các biểu đồ sẽ được tự động lưu dưới dạng tệp PDF trong thư mục **plots**.

<!-- ### Thuật toán tối đa hóa mô-đun phụ từ "Best of Both Worlds: Practical and Theoretically Optimal Submodule Maximization in Parallel": ### -->

**LS+PGB**.

-   **LinearSeq()** [**LS**] -- chạy thuật toán _LINEARSEQ_ cho SMCC (Thuật toán 1). Quy trình này triển khai các tối ưu hóa được mô tả trong _Phần G_.
-   **parallel_threshold_sample()** [TS] -- chạy thuật toán Parallelizable Greedy Algorithm _THRESHOLDSEQ_ cho ngưỡng cố định 'tau' (Thuật toán 3)
-   **ParallelGreedyBoost()** [**LS+PGB**] -- chạy _PARALLELGREEDYBOOST_ để Boost tới Tỷ lệ tối ưu (Thuật toán 4) bằng cách sử dụng _alpha_ và _gamma_ thu được bằng cách chạy _LinearSeq()_

Tất cả các quy trình được triển khai trong **_src/submodular_PGB.py_**
