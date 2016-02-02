classdef gcsparse < handle
    % sparse array GPU class
    % Usage:
    % A=gcsparse(A,[format]);
    % A=gcsparse(col,row,val,[nrows,[ncols,[format]]]);
    % format: 0 for COO, 1 for CSR (0 is default);
    % A: can be  matlab full/sparse array or gcsparse itself
    %
    % overloaded operators:
    % transpose: B=A.';
    % transpose: B=A';
    % multiply: x=A*y; (spmv)
    % size: [row, columns]
    % type: class/real/complex
    %
    % format conversion:
    %          B=real(A);
    %          A=complex(B);
    %          B=gcsparse(A,format);
    %          rowptr= ptr2row(A);
    %          row   =grow2ptr(A);
    % row <-> offset pointer conversion may crash inside the function,
    % but manually does not:
    %       so, to convert from A COO, to B CSR one can use this instead:
    %          B=A; %copy
    %          B.row= gptr2row(A.row,int32(A.nrows+1),A.nnz);
    %          B.format=1;
    %
    % S. Marchesini, LBNL 2013
    
    
    %    properties (SetAccess='private')
    properties
        nrows=int32(0); % number of rows
        ncols=int32(0); % number of columns
        nnz=int32(0);   % non zero elements
        val=gpuArray([]); %values  (gpu real/complex, single)
        col=gpuArray(int32([])); % column index (gpu int32)
        row=gpuArray(int32([])); % row/ptr index (gpu int32)
        format=int32(0); %0 for COO 1 for CSR
    end
    methods (Access = private)
    end
    methods (Static)
    end
    methods
        function obj = gcsparse(col,row,val,nrows,ncols,format)

            if nargin<6         %default is COO
                format=int32(0); %COO
            else
                format=int32(format);
            end
            
            if (nargin<=2); %gcsparse(A,[format])
                % get the sparse structure of A
                if nargin==2    %gcsparse(A,format) (format=row, second input)
                    format=int32(row); %row is actually the second input
                else
                    format=0;
                end
                
                if isa(col,'gcsparse') % we are just converting here
                      obj=col; %col is actually the fisrt input
                    if obj.format==format %nothing to do...
                        return
                    elseif (obj.format==0 && format==1) 
                         obj.row=row2ptr(obj); %COO->CSR
                        obj.format=format;
                    elseif (obj.format==1 && format==0);
                         %nptr=obj.nrows+1;
                         %rowptr= gptr2row(obj.row,nptr,obj.nnz);
                                                 rowptr=ptr2row(col);  %CSR-> COO
                         %                        obj.row=gptr2row(obj.row,nptr,obj.nnz);
                         obj.row=rowptr;
                         obj.format=format;
                    else
                        fprintf('invalid');
                    end
                    return
                else
                    % get  val,col,row triplets from A (first input)
                    [obj.ncols,obj.nrows]=size(col); %col is actually the fisrt input
                    obj.nrows=gather(obj.nrows);
                    obj.ncols=gather(obj.ncols);
                    
                    [obj.row,obj.col,obj.val]=find(col);
                    
                    obj.col=gpuArray(int32(obj.col(:)));
                    obj.row=gpuArray(int32(obj.row(:)));
                    obj.val=gpuArray((single(obj.val(:))));
                end
                if nargin==2    
                    format=int32(row); %row is actually the second input
                end
            else
                obj.col=gpuArray(int32(col(:)));
                obj.row=gpuArray(int32(row(:)));
                obj.val=gpuArray(val(:));
                obj.nrows=gather(int32(max(obj.row(:))));
                obj.ncols=gather(int32(max(obj.col(:))));
                obj.nnz=int32(obj.nnz);
            end
            
            obj.nnz=gather(int32(numel(obj.val)));
            
            % matlab to c indexing...:
            obj.col=obj.col-1;
            obj.row=obj.row-1;
            % increase nrows if input [nrows] is given
            if nargin>3
                if(~isempty(nrows))
                    obj.nrows=gather(int32(max(obj.nrows,nrows)));
                end
                if nargin>4
                    if (~isempty(ncols))
                        obj.ncols=int32(max(obj.ncols,ncols));
                    end
                end
            end
            % sort by rows
            [obj.row,unsort2sort]=sort(obj.row);
            obj.col=obj.col(unsort2sort);
            obj.val=obj.val(unsort2sort);
            obj.format=0;
            
            if  format==1;
                %                 obj.row=coo2csr(obj);
                obj.row= row2ptr(obj);
                obj.format=1;
            end
            %                'hi'
        end
        function B=real(A)
            B=A;
            B.val=real(B.val);
        end
        function B=complex(A)
            B=A;
            if isreal(A.val);
                B.val=complex(B.val);
            end
        end
        function y = mtimes(A,x) %SpMV
            %SpMV with CUSP
            if A.format==0
                %                 y=0;
                %                'todo'
                y=gspmv_coo(A.val,A.col,A.row,A.nrows, x);
            elseif A.format==1
                %            y=gspmv_csr(A.col,A.row,A.val,A.nrows,A.ncols,x);
                y=gspmv_csr(A.val,A.col,A.row,x);
            end
        end
        function C= ctranspose(obj)
            % format->coo->transpose->format
            C=gcsparse(obj,0); %convert to COO
            tmprow=C.col; %swap row and columns
            C.col=C.row;
            C.row=tmprow;
            tmp=C.nrows;
            C.nrows=obj.ncols;
            C.ncols=tmp;
            C.val=conj(obj.val); %conjugate
            C=gcsparse(C,obj.format); %revert to original format
        end
        
        function C= transpose(obj)
            C=gcsparse(obj,0); %convert to COO
            tmprow=C.col;  %swap row and columns
            C.col=C.row;
            C.row=tmprow;
            tmp=C.nrows;
            C.nrows=obj.ncols;
            C.ncols=tmp;
            C=gcsparse(C,obj.format);
        end
        function [row,col,val]= find(obj)
           if obj.format==1;
                fprintf('it may not work, use COO\n')
                fprintf('[col,row,val]=find(gcsparse(A,0))');
%                 [~,row,~]=find(gcsparse(A,0));
                nptr=int32(obj.nrows+1);
                ptr=obj.row+0;
                nnz=obj.nnz+0;
                row=gptr2row(ptr,nptr,nnz);
%                row=ptr2row(obj);
                row=row+1;
                if numel(row)<obj.nnz
                    fprintf('did not work, use COO\n')
                end
%                 row=gptr2row(obj.row,int32(obj.nrows+1),obj.nnz);
           else
                row=obj.row+1;
           end
           col=obj.col+1;
           val=obj.val;

        end
          
                    
        function m = size(obj)
            m=[obj.nrows obj.ncols];
        end
        function m = type(obj)
            f0= classUnderlying(obj.val);
            if (isreal(obj.val))
                fmt='Real';
            else fmt='Complex';
            end
            m=[f0 ' ' fmt];
        end
        function row= ptr2row(obj)
            %  offset pointer to row conversion
            row= gptr2row(obj.row,int32(obj.nrows+1),obj.nnz);
        end
        function rowptr= row2ptr(obj)
            %  row to offsets
            rowptr=grow2ptr(obj.row,(obj.nrows+1),(obj.nnz));
        end
        
        %          function c = plus(a,b)
        %          % todo       fastadd(a,b,c);
        %                 return
        %         end
        %         function b = subsref(A, S)
        %         end
        %         function c = find(a)
        %         end
        %          function c = find(a)
        %          end
    end
    
end
