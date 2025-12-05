FROM python:3.9-slim 

# Install system dependencies 
RUN apt-get update && apt-get install -y \ 
	build-essential \ 
	gcc \ 
	g++ \ 
	cmake \ 
	libgsl-dev \ 
	git \ 
	pybind11-dev \
	&& rm -rf /var/lib/apt/lists/* 
	
# Install Python dependencies 
RUN pip install --no-cache-dir \  
	numpy \ 
	pandas \ 
	matplotlib \ 
	scipy \ 
	jupyter \ 
	tqdm \ 
	scikit-optimize \ 
	scikit-learn 
	
# Set working directory 
WORKDIR /workspace 

# Copy project files 
COPY . /workspace/ 

# Build the C++ functions 
WORKDIR /workspace/func_build 
RUN cmake CMakeLists.txt 
RUN cmake --build . 

# Return to main workspace 
WORKDIR /workspace 

# Default command to run Jupyter Notebook
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]

