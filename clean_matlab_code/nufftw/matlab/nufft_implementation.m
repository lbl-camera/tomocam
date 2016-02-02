% NUFFT_IMPLEMENTATION

% Michal Zarrouk, July 2013.

classdef nufft_implementation < handle
    properties %(SetAccess = private, Hidden = true)
        objectHandle; % Handle to the underlying nuFFT_implementation_t instance - a pointer to the C++ nuFFT_implemetation_t instance
        datatype;
        direction = 'forward';
        Nthreads = 1;
        dontdelete = 0;
    end
    methods
        %% Constructor - Create a new nuFFT_implementation
        function this = nufft_implementation(varargin)
            % NUFFT_IMPLEMENTATION - class constructor
            %   imp = nufft_implementation(datatype, Nthreads, imagesize,�maxerr, alpha, resamplingmethod, trajfilename);
            %   imp = nufft_implementation(datatype, Nthreads, imagesize,�maxerr, alpha, resamplingmethod, trajectory, sqrtdcf)
            %   imp = nufft_implementation(datatype, Nthreads, impfilename);
            %   imp = nufft_implementation(datatype, pointer_to_nuFFT_implementation_t);
            if nargin > 0
                this.datatype = varargin{1};
                if nargin == 2
                    pointer_to_nuFFT_implementation_t = varargin{2};
                    this.objectHandle = pointer_to_nuFFT_implementation_t;
                else
                    this.Nthreads = varargin{2};
                    this.objectHandle = nufft_mex(this, 'init', varargin{2:end});
                end
            end
        end
        
        %% Destructor - Destroy the nufft_implementation class instance
        function delete(this)
            % DELETE - class destructor
            %   imp.delete;
            if ~this.dontdelete
                nufft_mex(this, 'delete', this.objectHandle);    
            end
        end

        %% forward
        function kdata = forward(this, Nthreads, image_data)
            % FORWARD - Forward nuFFT
            %   kdata = imp.forward(Nthreads, image_data);
			kdata = nufft_mex(this, 'forward', Nthreads, this.objectHandle, image_data);
        end
        
        %% adjoint
        function image_data = adjoint(this, Nthreads, kdata)
            % ADJOINT - Adjoint nuFFT
            %   image_data = imp.adjoint(Nthreads, kdata);
            image_data = nufft_mex(this, 'adjoint', Nthreads, this.objectHandle, kdata);
        end
        
        %% create_impfile
        function create_impfile(this, impfilename)
            % CREATE_IMPFILE - Write nuFFT implementation to an implementation file
            %   imp.create_impfile(impfilename);
            nufft_mex(this, 'create_impfile', this.objectHandle, impfilename);
        end
        
        function res = mtimes(this, data)
            if strcmp(this.direction,'forward')
                res = this.forward(this.Nthreads,data);
            elseif strcmp(this.direction,'adjoint')
                res = this.adjoint(this.Nthreads,data);
            end
        end
        
        function res = ctranspose(this)
            res = copy(this);
            if strcmp(res.direction,'forward')
                res.direction = 'adjoint';
            elseif strcmp(res.direction,'adjoint')
                res.direction = 'forward';
            end
        end
        
        
        function res = copy(this)
            res = nufft_implementation;
            a = properties(this);
            for i=1:length(a)
                res.(a{i}) = this.(a{i});
            end
            res.dontdelete = 1;
        end
        
    end
end


function varargout = nufft_mex(this, varargin)
    if strcmp(this.datatype,'double')
        if nargout == 0
            nufft_mex_double(varargin{:});
            varargout = {};
        else
            varargout{:} = nufft_mex_double(varargin{:});
        end
    elseif strcmp(this.datatype,'float')
        if nargout == 0
            nufft_mex_float(varargin{:});
            varargout = {};
        else
            varargout{:} = nufft_mex_float(varargin{:});
        end
    end

end
