È¬
Ý
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68µ¸

hiddel_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*$
shared_namehiddel_layer/kernel
|
'hiddel_layer/kernel/Read/ReadVariableOpReadVariableOphiddel_layer/kernel*
_output_shapes
:	*
dtype0
{
hiddel_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namehiddel_layer/bias
t
%hiddel_layer/bias/Read/ReadVariableOpReadVariableOphiddel_layer/bias*
_output_shapes	
:*
dtype0

hiddel_layer_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_namehiddel_layer_2/kernel

)hiddel_layer_2/kernel/Read/ReadVariableOpReadVariableOphiddel_layer_2/kernel* 
_output_shapes
:
*
dtype0

hiddel_layer_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namehiddel_layer_2/bias
x
'hiddel_layer_2/bias/Read/ReadVariableOpReadVariableOphiddel_layer_2/bias*
_output_shapes	
:*
dtype0

hiddel_layer_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_namehiddel_layer_3/kernel

)hiddel_layer_3/kernel/Read/ReadVariableOpReadVariableOphiddel_layer_3/kernel* 
_output_shapes
:
*
dtype0

hiddel_layer_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namehiddel_layer_3/bias
x
'hiddel_layer_3/bias/Read/ReadVariableOpReadVariableOphiddel_layer_3/bias*
_output_shapes	
:*
dtype0

prediction/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*"
shared_nameprediction/kernel
x
%prediction/kernel/Read/ReadVariableOpReadVariableOpprediction/kernel*
_output_shapes
:	*
dtype0
v
prediction/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameprediction/bias
o
#prediction/bias/Read/ReadVariableOpReadVariableOpprediction/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/hiddel_layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*+
shared_nameAdam/hiddel_layer/kernel/m

.Adam/hiddel_layer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hiddel_layer/kernel/m*
_output_shapes
:	*
dtype0

Adam/hiddel_layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/hiddel_layer/bias/m

,Adam/hiddel_layer/bias/m/Read/ReadVariableOpReadVariableOpAdam/hiddel_layer/bias/m*
_output_shapes	
:*
dtype0

Adam/hiddel_layer_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*-
shared_nameAdam/hiddel_layer_2/kernel/m

0Adam/hiddel_layer_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hiddel_layer_2/kernel/m* 
_output_shapes
:
*
dtype0

Adam/hiddel_layer_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/hiddel_layer_2/bias/m

.Adam/hiddel_layer_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/hiddel_layer_2/bias/m*
_output_shapes	
:*
dtype0

Adam/hiddel_layer_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*-
shared_nameAdam/hiddel_layer_3/kernel/m

0Adam/hiddel_layer_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hiddel_layer_3/kernel/m* 
_output_shapes
:
*
dtype0

Adam/hiddel_layer_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/hiddel_layer_3/bias/m

.Adam/hiddel_layer_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/hiddel_layer_3/bias/m*
_output_shapes	
:*
dtype0

Adam/prediction/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameAdam/prediction/kernel/m

,Adam/prediction/kernel/m/Read/ReadVariableOpReadVariableOpAdam/prediction/kernel/m*
_output_shapes
:	*
dtype0

Adam/prediction/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/prediction/bias/m
}
*Adam/prediction/bias/m/Read/ReadVariableOpReadVariableOpAdam/prediction/bias/m*
_output_shapes
:*
dtype0

Adam/hiddel_layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*+
shared_nameAdam/hiddel_layer/kernel/v

.Adam/hiddel_layer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hiddel_layer/kernel/v*
_output_shapes
:	*
dtype0

Adam/hiddel_layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/hiddel_layer/bias/v

,Adam/hiddel_layer/bias/v/Read/ReadVariableOpReadVariableOpAdam/hiddel_layer/bias/v*
_output_shapes	
:*
dtype0

Adam/hiddel_layer_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*-
shared_nameAdam/hiddel_layer_2/kernel/v

0Adam/hiddel_layer_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hiddel_layer_2/kernel/v* 
_output_shapes
:
*
dtype0

Adam/hiddel_layer_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/hiddel_layer_2/bias/v

.Adam/hiddel_layer_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/hiddel_layer_2/bias/v*
_output_shapes	
:*
dtype0

Adam/hiddel_layer_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*-
shared_nameAdam/hiddel_layer_3/kernel/v

0Adam/hiddel_layer_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hiddel_layer_3/kernel/v* 
_output_shapes
:
*
dtype0

Adam/hiddel_layer_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/hiddel_layer_3/bias/v

.Adam/hiddel_layer_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/hiddel_layer_3/bias/v*
_output_shapes	
:*
dtype0

Adam/prediction/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameAdam/prediction/kernel/v

,Adam/prediction/kernel/v/Read/ReadVariableOpReadVariableOpAdam/prediction/kernel/v*
_output_shapes
:	*
dtype0

Adam/prediction/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/prediction/bias/v
}
*Adam/prediction/bias/v/Read/ReadVariableOpReadVariableOpAdam/prediction/bias/v*
_output_shapes
:*
dtype0

Adam/hiddel_layer/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*.
shared_nameAdam/hiddel_layer/kernel/vhat

1Adam/hiddel_layer/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/hiddel_layer/kernel/vhat*
_output_shapes
:	*
dtype0

Adam/hiddel_layer/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/hiddel_layer/bias/vhat

/Adam/hiddel_layer/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/hiddel_layer/bias/vhat*
_output_shapes	
:*
dtype0

Adam/hiddel_layer_2/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*0
shared_name!Adam/hiddel_layer_2/kernel/vhat

3Adam/hiddel_layer_2/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/hiddel_layer_2/kernel/vhat* 
_output_shapes
:
*
dtype0

Adam/hiddel_layer_2/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/hiddel_layer_2/bias/vhat

1Adam/hiddel_layer_2/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/hiddel_layer_2/bias/vhat*
_output_shapes	
:*
dtype0

Adam/hiddel_layer_3/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*0
shared_name!Adam/hiddel_layer_3/kernel/vhat

3Adam/hiddel_layer_3/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/hiddel_layer_3/kernel/vhat* 
_output_shapes
:
*
dtype0

Adam/hiddel_layer_3/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/hiddel_layer_3/bias/vhat

1Adam/hiddel_layer_3/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/hiddel_layer_3/bias/vhat*
_output_shapes	
:*
dtype0

Adam/prediction/kernel/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*,
shared_nameAdam/prediction/kernel/vhat

/Adam/prediction/kernel/vhat/Read/ReadVariableOpReadVariableOpAdam/prediction/kernel/vhat*
_output_shapes
:	*
dtype0

Adam/prediction/bias/vhatVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/prediction/bias/vhat

-Adam/prediction/bias/vhat/Read/ReadVariableOpReadVariableOpAdam/prediction/bias/vhat*
_output_shapes
:*
dtype0

NoOpNoOp
¯B
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*êA
valueàABÝA BÖA
õ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
¦

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses*
¦

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses*
´
/iter

0beta_1

1beta_2
	2decay
3learning_ratemYmZm[m\m] m^'m_(m`vavbvcvdve vf'vg(vh
vhati
vhatj
vhatk
vhatl
vhatm
 vhatn
'vhato
(vhatp*
<
0
1
2
3
4
 5
'6
(7*
<
0
1
2
3
4
 5
'6
(7*
* 
°
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

9serving_default* 
c]
VARIABLE_VALUEhiddel_layer/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEhiddel_layer/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
e_
VARIABLE_VALUEhiddel_layer_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEhiddel_layer_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
e_
VARIABLE_VALUEhiddel_layer_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEhiddel_layer_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
 1*

0
 1*
* 

Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*
* 
* 
a[
VARIABLE_VALUEprediction/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEprediction/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

'0
(1*

'0
(1*
* 

Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
2
3
4*

N0
O1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
	Ptotal
	Qcount
R	variables
S	keras_api*
H
	Ttotal
	Ucount
V
_fn_kwargs
W	variables
X	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

P0
Q1*

R	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

T0
U1*

W	variables*

VARIABLE_VALUEAdam/hiddel_layer/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/hiddel_layer/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/hiddel_layer_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/hiddel_layer_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/hiddel_layer_3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/hiddel_layer_3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/prediction/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/prediction/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/hiddel_layer/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/hiddel_layer/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/hiddel_layer_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/hiddel_layer_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/hiddel_layer_3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/hiddel_layer_3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/prediction/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/prediction/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/hiddel_layer/kernel/vhatUlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/hiddel_layer/bias/vhatSlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/hiddel_layer_2/kernel/vhatUlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/hiddel_layer_2/bias/vhatSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/hiddel_layer_3/kernel/vhatUlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/hiddel_layer_3/bias/vhatSlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/prediction/kernel/vhatUlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/prediction/bias/vhatSlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUE*
x
serving_default_inputPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
å
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputhiddel_layer/kernelhiddel_layer/biashiddel_layer_2/kernelhiddel_layer_2/biashiddel_layer_3/kernelhiddel_layer_3/biasprediction/kernelprediction/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_2950250
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
®
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'hiddel_layer/kernel/Read/ReadVariableOp%hiddel_layer/bias/Read/ReadVariableOp)hiddel_layer_2/kernel/Read/ReadVariableOp'hiddel_layer_2/bias/Read/ReadVariableOp)hiddel_layer_3/kernel/Read/ReadVariableOp'hiddel_layer_3/bias/Read/ReadVariableOp%prediction/kernel/Read/ReadVariableOp#prediction/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp.Adam/hiddel_layer/kernel/m/Read/ReadVariableOp,Adam/hiddel_layer/bias/m/Read/ReadVariableOp0Adam/hiddel_layer_2/kernel/m/Read/ReadVariableOp.Adam/hiddel_layer_2/bias/m/Read/ReadVariableOp0Adam/hiddel_layer_3/kernel/m/Read/ReadVariableOp.Adam/hiddel_layer_3/bias/m/Read/ReadVariableOp,Adam/prediction/kernel/m/Read/ReadVariableOp*Adam/prediction/bias/m/Read/ReadVariableOp.Adam/hiddel_layer/kernel/v/Read/ReadVariableOp,Adam/hiddel_layer/bias/v/Read/ReadVariableOp0Adam/hiddel_layer_2/kernel/v/Read/ReadVariableOp.Adam/hiddel_layer_2/bias/v/Read/ReadVariableOp0Adam/hiddel_layer_3/kernel/v/Read/ReadVariableOp.Adam/hiddel_layer_3/bias/v/Read/ReadVariableOp,Adam/prediction/kernel/v/Read/ReadVariableOp*Adam/prediction/bias/v/Read/ReadVariableOp1Adam/hiddel_layer/kernel/vhat/Read/ReadVariableOp/Adam/hiddel_layer/bias/vhat/Read/ReadVariableOp3Adam/hiddel_layer_2/kernel/vhat/Read/ReadVariableOp1Adam/hiddel_layer_2/bias/vhat/Read/ReadVariableOp3Adam/hiddel_layer_3/kernel/vhat/Read/ReadVariableOp1Adam/hiddel_layer_3/bias/vhat/Read/ReadVariableOp/Adam/prediction/kernel/vhat/Read/ReadVariableOp-Adam/prediction/bias/vhat/Read/ReadVariableOpConst*6
Tin/
-2+	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_save_2950476
õ	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamehiddel_layer/kernelhiddel_layer/biashiddel_layer_2/kernelhiddel_layer_2/biashiddel_layer_3/kernelhiddel_layer_3/biasprediction/kernelprediction/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/hiddel_layer/kernel/mAdam/hiddel_layer/bias/mAdam/hiddel_layer_2/kernel/mAdam/hiddel_layer_2/bias/mAdam/hiddel_layer_3/kernel/mAdam/hiddel_layer_3/bias/mAdam/prediction/kernel/mAdam/prediction/bias/mAdam/hiddel_layer/kernel/vAdam/hiddel_layer/bias/vAdam/hiddel_layer_2/kernel/vAdam/hiddel_layer_2/bias/vAdam/hiddel_layer_3/kernel/vAdam/hiddel_layer_3/bias/vAdam/prediction/kernel/vAdam/prediction/bias/vAdam/hiddel_layer/kernel/vhatAdam/hiddel_layer/bias/vhatAdam/hiddel_layer_2/kernel/vhatAdam/hiddel_layer_2/bias/vhatAdam/hiddel_layer_3/kernel/vhatAdam/hiddel_layer_3/bias/vhatAdam/prediction/kernel/vhatAdam/prediction/bias/vhat*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__traced_restore_2950609
Î
À
G__inference_full_model_layer_call_and_return_conditional_losses_2950091	
input'
hiddel_layer_2950070:	#
hiddel_layer_2950072:	*
hiddel_layer_2_2950075:
%
hiddel_layer_2_2950077:	*
hiddel_layer_3_2950080:
%
hiddel_layer_3_2950082:	%
prediction_2950085:	 
prediction_2950087:
identity¢$hiddel_layer/StatefulPartitionedCall¢&hiddel_layer_2/StatefulPartitionedCall¢&hiddel_layer_3/StatefulPartitionedCall¢"prediction/StatefulPartitionedCall
$hiddel_layer/StatefulPartitionedCallStatefulPartitionedCallinputhiddel_layer_2950070hiddel_layer_2950072*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hiddel_layer_layer_call_and_return_conditional_losses_2949863³
&hiddel_layer_2/StatefulPartitionedCallStatefulPartitionedCall-hiddel_layer/StatefulPartitionedCall:output:0hiddel_layer_2_2950075hiddel_layer_2_2950077*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hiddel_layer_2_layer_call_and_return_conditional_losses_2949880µ
&hiddel_layer_3/StatefulPartitionedCallStatefulPartitionedCall/hiddel_layer_2/StatefulPartitionedCall:output:0hiddel_layer_3_2950080hiddel_layer_3_2950082*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hiddel_layer_3_layer_call_and_return_conditional_losses_2949897¤
"prediction/StatefulPartitionedCallStatefulPartitionedCall/hiddel_layer_3/StatefulPartitionedCall:output:0prediction_2950085prediction_2950087*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_prediction_layer_call_and_return_conditional_losses_2949914z
IdentityIdentity+prediction/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿä
NoOpNoOp%^hiddel_layer/StatefulPartitionedCall'^hiddel_layer_2/StatefulPartitionedCall'^hiddel_layer_3/StatefulPartitionedCall#^prediction/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2L
$hiddel_layer/StatefulPartitionedCall$hiddel_layer/StatefulPartitionedCall2P
&hiddel_layer_2/StatefulPartitionedCall&hiddel_layer_2/StatefulPartitionedCall2P
&hiddel_layer_3/StatefulPartitionedCall&hiddel_layer_3/StatefulPartitionedCall2H
"prediction/StatefulPartitionedCall"prediction/StatefulPartitionedCall:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
Ñ
Á
G__inference_full_model_layer_call_and_return_conditional_losses_2950027

inputs'
hiddel_layer_2950006:	#
hiddel_layer_2950008:	*
hiddel_layer_2_2950011:
%
hiddel_layer_2_2950013:	*
hiddel_layer_3_2950016:
%
hiddel_layer_3_2950018:	%
prediction_2950021:	 
prediction_2950023:
identity¢$hiddel_layer/StatefulPartitionedCall¢&hiddel_layer_2/StatefulPartitionedCall¢&hiddel_layer_3/StatefulPartitionedCall¢"prediction/StatefulPartitionedCall
$hiddel_layer/StatefulPartitionedCallStatefulPartitionedCallinputshiddel_layer_2950006hiddel_layer_2950008*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hiddel_layer_layer_call_and_return_conditional_losses_2949863³
&hiddel_layer_2/StatefulPartitionedCallStatefulPartitionedCall-hiddel_layer/StatefulPartitionedCall:output:0hiddel_layer_2_2950011hiddel_layer_2_2950013*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hiddel_layer_2_layer_call_and_return_conditional_losses_2949880µ
&hiddel_layer_3/StatefulPartitionedCallStatefulPartitionedCall/hiddel_layer_2/StatefulPartitionedCall:output:0hiddel_layer_3_2950016hiddel_layer_3_2950018*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hiddel_layer_3_layer_call_and_return_conditional_losses_2949897¤
"prediction/StatefulPartitionedCallStatefulPartitionedCall/hiddel_layer_3/StatefulPartitionedCall:output:0prediction_2950021prediction_2950023*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_prediction_layer_call_and_return_conditional_losses_2949914z
IdentityIdentity+prediction/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿä
NoOpNoOp%^hiddel_layer/StatefulPartitionedCall'^hiddel_layer_2/StatefulPartitionedCall'^hiddel_layer_3/StatefulPartitionedCall#^prediction/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2L
$hiddel_layer/StatefulPartitionedCall$hiddel_layer/StatefulPartitionedCall2P
&hiddel_layer_2/StatefulPartitionedCall&hiddel_layer_2/StatefulPartitionedCall2P
&hiddel_layer_3/StatefulPartitionedCall&hiddel_layer_3/StatefulPartitionedCall2H
"prediction/StatefulPartitionedCall"prediction/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
 
0__inference_hiddel_layer_2_layer_call_fn_2950279

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hiddel_layer_2_layer_call_and_return_conditional_losses_2949880p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð

.__inference_hiddel_layer_layer_call_fn_2950259

inputs
unknown:	
	unknown_0:	
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hiddel_layer_layer_call_and_return_conditional_losses_2949863p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í	
Ä
,__inference_full_model_layer_call_fn_2950163

inputs
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_full_model_layer_call_and_return_conditional_losses_2950027o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
¼
%__inference_signature_wrapper_2950250	
input
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_2949845o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
/

"__inference__wrapped_model_2949845	
inputI
6full_model_hiddel_layer_matmul_readvariableop_resource:	F
7full_model_hiddel_layer_biasadd_readvariableop_resource:	L
8full_model_hiddel_layer_2_matmul_readvariableop_resource:
H
9full_model_hiddel_layer_2_biasadd_readvariableop_resource:	L
8full_model_hiddel_layer_3_matmul_readvariableop_resource:
H
9full_model_hiddel_layer_3_biasadd_readvariableop_resource:	G
4full_model_prediction_matmul_readvariableop_resource:	C
5full_model_prediction_biasadd_readvariableop_resource:
identity¢.full_model/hiddel_layer/BiasAdd/ReadVariableOp¢-full_model/hiddel_layer/MatMul/ReadVariableOp¢0full_model/hiddel_layer_2/BiasAdd/ReadVariableOp¢/full_model/hiddel_layer_2/MatMul/ReadVariableOp¢0full_model/hiddel_layer_3/BiasAdd/ReadVariableOp¢/full_model/hiddel_layer_3/MatMul/ReadVariableOp¢,full_model/prediction/BiasAdd/ReadVariableOp¢+full_model/prediction/MatMul/ReadVariableOp¥
-full_model/hiddel_layer/MatMul/ReadVariableOpReadVariableOp6full_model_hiddel_layer_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
full_model/hiddel_layer/MatMulMatMulinput5full_model/hiddel_layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
.full_model/hiddel_layer/BiasAdd/ReadVariableOpReadVariableOp7full_model_hiddel_layer_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¿
full_model/hiddel_layer/BiasAddBiasAdd(full_model/hiddel_layer/MatMul:product:06full_model/hiddel_layer/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
full_model/hiddel_layer/ReluRelu(full_model/hiddel_layer/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
/full_model/hiddel_layer_2/MatMul/ReadVariableOpReadVariableOp8full_model_hiddel_layer_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Â
 full_model/hiddel_layer_2/MatMulMatMul*full_model/hiddel_layer/Relu:activations:07full_model/hiddel_layer_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
0full_model/hiddel_layer_2/BiasAdd/ReadVariableOpReadVariableOp9full_model_hiddel_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Å
!full_model/hiddel_layer_2/BiasAddBiasAdd*full_model/hiddel_layer_2/MatMul:product:08full_model/hiddel_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
full_model/hiddel_layer_2/ReluRelu*full_model/hiddel_layer_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
/full_model/hiddel_layer_3/MatMul/ReadVariableOpReadVariableOp8full_model_hiddel_layer_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Ä
 full_model/hiddel_layer_3/MatMulMatMul,full_model/hiddel_layer_2/Relu:activations:07full_model/hiddel_layer_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
0full_model/hiddel_layer_3/BiasAdd/ReadVariableOpReadVariableOp9full_model_hiddel_layer_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Å
!full_model/hiddel_layer_3/BiasAddBiasAdd*full_model/hiddel_layer_3/MatMul:product:08full_model/hiddel_layer_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
full_model/hiddel_layer_3/ReluRelu*full_model/hiddel_layer_3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
+full_model/prediction/MatMul/ReadVariableOpReadVariableOp4full_model_prediction_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0»
full_model/prediction/MatMulMatMul,full_model/hiddel_layer_3/Relu:activations:03full_model/prediction/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,full_model/prediction/BiasAdd/ReadVariableOpReadVariableOp5full_model_prediction_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¸
full_model/prediction/BiasAddBiasAdd&full_model/prediction/MatMul:product:04full_model/prediction/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
full_model/prediction/SoftmaxSoftmax&full_model/prediction/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
IdentityIdentity'full_model/prediction/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÎ
NoOpNoOp/^full_model/hiddel_layer/BiasAdd/ReadVariableOp.^full_model/hiddel_layer/MatMul/ReadVariableOp1^full_model/hiddel_layer_2/BiasAdd/ReadVariableOp0^full_model/hiddel_layer_2/MatMul/ReadVariableOp1^full_model/hiddel_layer_3/BiasAdd/ReadVariableOp0^full_model/hiddel_layer_3/MatMul/ReadVariableOp-^full_model/prediction/BiasAdd/ReadVariableOp,^full_model/prediction/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2`
.full_model/hiddel_layer/BiasAdd/ReadVariableOp.full_model/hiddel_layer/BiasAdd/ReadVariableOp2^
-full_model/hiddel_layer/MatMul/ReadVariableOp-full_model/hiddel_layer/MatMul/ReadVariableOp2d
0full_model/hiddel_layer_2/BiasAdd/ReadVariableOp0full_model/hiddel_layer_2/BiasAdd/ReadVariableOp2b
/full_model/hiddel_layer_2/MatMul/ReadVariableOp/full_model/hiddel_layer_2/MatMul/ReadVariableOp2d
0full_model/hiddel_layer_3/BiasAdd/ReadVariableOp0full_model/hiddel_layer_3/BiasAdd/ReadVariableOp2b
/full_model/hiddel_layer_3/MatMul/ReadVariableOp/full_model/hiddel_layer_3/MatMul/ReadVariableOp2\
,full_model/prediction/BiasAdd/ReadVariableOp,full_model/prediction/BiasAdd/ReadVariableOp2Z
+full_model/prediction/MatMul/ReadVariableOp+full_model/prediction/MatMul/ReadVariableOp:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
®

ÿ
K__inference_hiddel_layer_3_layer_call_and_return_conditional_losses_2949897

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë

,__inference_prediction_layer_call_fn_2950319

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_prediction_layer_call_and_return_conditional_losses_2949914o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
 
0__inference_hiddel_layer_3_layer_call_fn_2950299

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hiddel_layer_3_layer_call_and_return_conditional_losses_2949897p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®

ÿ
K__inference_hiddel_layer_2_layer_call_and_return_conditional_losses_2950290

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨

ü
I__inference_hiddel_layer_layer_call_and_return_conditional_losses_2950270

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®

ÿ
K__inference_hiddel_layer_3_layer_call_and_return_conditional_losses_2950310

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®

ÿ
K__inference_hiddel_layer_2_layer_call_and_return_conditional_losses_2949880

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï'

G__inference_full_model_layer_call_and_return_conditional_losses_2950227

inputs>
+hiddel_layer_matmul_readvariableop_resource:	;
,hiddel_layer_biasadd_readvariableop_resource:	A
-hiddel_layer_2_matmul_readvariableop_resource:
=
.hiddel_layer_2_biasadd_readvariableop_resource:	A
-hiddel_layer_3_matmul_readvariableop_resource:
=
.hiddel_layer_3_biasadd_readvariableop_resource:	<
)prediction_matmul_readvariableop_resource:	8
*prediction_biasadd_readvariableop_resource:
identity¢#hiddel_layer/BiasAdd/ReadVariableOp¢"hiddel_layer/MatMul/ReadVariableOp¢%hiddel_layer_2/BiasAdd/ReadVariableOp¢$hiddel_layer_2/MatMul/ReadVariableOp¢%hiddel_layer_3/BiasAdd/ReadVariableOp¢$hiddel_layer_3/MatMul/ReadVariableOp¢!prediction/BiasAdd/ReadVariableOp¢ prediction/MatMul/ReadVariableOp
"hiddel_layer/MatMul/ReadVariableOpReadVariableOp+hiddel_layer_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
hiddel_layer/MatMulMatMulinputs*hiddel_layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#hiddel_layer/BiasAdd/ReadVariableOpReadVariableOp,hiddel_layer_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
hiddel_layer/BiasAddBiasAddhiddel_layer/MatMul:product:0+hiddel_layer/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
hiddel_layer/ReluReluhiddel_layer/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$hiddel_layer_2/MatMul/ReadVariableOpReadVariableOp-hiddel_layer_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0¡
hiddel_layer_2/MatMulMatMulhiddel_layer/Relu:activations:0,hiddel_layer_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%hiddel_layer_2/BiasAdd/ReadVariableOpReadVariableOp.hiddel_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¤
hiddel_layer_2/BiasAddBiasAddhiddel_layer_2/MatMul:product:0-hiddel_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
hiddel_layer_2/ReluReluhiddel_layer_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$hiddel_layer_3/MatMul/ReadVariableOpReadVariableOp-hiddel_layer_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0£
hiddel_layer_3/MatMulMatMul!hiddel_layer_2/Relu:activations:0,hiddel_layer_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%hiddel_layer_3/BiasAdd/ReadVariableOpReadVariableOp.hiddel_layer_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¤
hiddel_layer_3/BiasAddBiasAddhiddel_layer_3/MatMul:product:0-hiddel_layer_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
hiddel_layer_3/ReluReluhiddel_layer_3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 prediction/MatMul/ReadVariableOpReadVariableOp)prediction_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
prediction/MatMulMatMul!hiddel_layer_3/Relu:activations:0(prediction/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!prediction/BiasAdd/ReadVariableOpReadVariableOp*prediction_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
prediction/BiasAddBiasAddprediction/MatMul:product:0)prediction/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
prediction/SoftmaxSoftmaxprediction/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
IdentityIdentityprediction/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿö
NoOpNoOp$^hiddel_layer/BiasAdd/ReadVariableOp#^hiddel_layer/MatMul/ReadVariableOp&^hiddel_layer_2/BiasAdd/ReadVariableOp%^hiddel_layer_2/MatMul/ReadVariableOp&^hiddel_layer_3/BiasAdd/ReadVariableOp%^hiddel_layer_3/MatMul/ReadVariableOp"^prediction/BiasAdd/ReadVariableOp!^prediction/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2J
#hiddel_layer/BiasAdd/ReadVariableOp#hiddel_layer/BiasAdd/ReadVariableOp2H
"hiddel_layer/MatMul/ReadVariableOp"hiddel_layer/MatMul/ReadVariableOp2N
%hiddel_layer_2/BiasAdd/ReadVariableOp%hiddel_layer_2/BiasAdd/ReadVariableOp2L
$hiddel_layer_2/MatMul/ReadVariableOp$hiddel_layer_2/MatMul/ReadVariableOp2N
%hiddel_layer_3/BiasAdd/ReadVariableOp%hiddel_layer_3/BiasAdd/ReadVariableOp2L
$hiddel_layer_3/MatMul/ReadVariableOp$hiddel_layer_3/MatMul/ReadVariableOp2F
!prediction/BiasAdd/ReadVariableOp!prediction/BiasAdd/ReadVariableOp2D
 prediction/MatMul/ReadVariableOp prediction/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®§

#__inference__traced_restore_2950609
file_prefix7
$assignvariableop_hiddel_layer_kernel:	3
$assignvariableop_1_hiddel_layer_bias:	<
(assignvariableop_2_hiddel_layer_2_kernel:
5
&assignvariableop_3_hiddel_layer_2_bias:	<
(assignvariableop_4_hiddel_layer_3_kernel:
5
&assignvariableop_5_hiddel_layer_3_bias:	7
$assignvariableop_6_prediction_kernel:	0
"assignvariableop_7_prediction_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: A
.assignvariableop_17_adam_hiddel_layer_kernel_m:	;
,assignvariableop_18_adam_hiddel_layer_bias_m:	D
0assignvariableop_19_adam_hiddel_layer_2_kernel_m:
=
.assignvariableop_20_adam_hiddel_layer_2_bias_m:	D
0assignvariableop_21_adam_hiddel_layer_3_kernel_m:
=
.assignvariableop_22_adam_hiddel_layer_3_bias_m:	?
,assignvariableop_23_adam_prediction_kernel_m:	8
*assignvariableop_24_adam_prediction_bias_m:A
.assignvariableop_25_adam_hiddel_layer_kernel_v:	;
,assignvariableop_26_adam_hiddel_layer_bias_v:	D
0assignvariableop_27_adam_hiddel_layer_2_kernel_v:
=
.assignvariableop_28_adam_hiddel_layer_2_bias_v:	D
0assignvariableop_29_adam_hiddel_layer_3_kernel_v:
=
.assignvariableop_30_adam_hiddel_layer_3_bias_v:	?
,assignvariableop_31_adam_prediction_kernel_v:	8
*assignvariableop_32_adam_prediction_bias_v:D
1assignvariableop_33_adam_hiddel_layer_kernel_vhat:	>
/assignvariableop_34_adam_hiddel_layer_bias_vhat:	G
3assignvariableop_35_adam_hiddel_layer_2_kernel_vhat:
@
1assignvariableop_36_adam_hiddel_layer_2_bias_vhat:	G
3assignvariableop_37_adam_hiddel_layer_3_kernel_vhat:
@
1assignvariableop_38_adam_hiddel_layer_3_bias_vhat:	B
/assignvariableop_39_adam_prediction_kernel_vhat:	;
-assignvariableop_40_adam_prediction_bias_vhat:
identity_42¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9â
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*
valueþBû*B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÄ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ó
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¾
_output_shapes«
¨::::::::::::::::::::::::::::::::::::::::::*8
dtypes.
,2*	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp$assignvariableop_hiddel_layer_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp$assignvariableop_1_hiddel_layer_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp(assignvariableop_2_hiddel_layer_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp&assignvariableop_3_hiddel_layer_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp(assignvariableop_4_hiddel_layer_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp&assignvariableop_5_hiddel_layer_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp$assignvariableop_6_prediction_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp"assignvariableop_7_prediction_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp.assignvariableop_17_adam_hiddel_layer_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp,assignvariableop_18_adam_hiddel_layer_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_19AssignVariableOp0assignvariableop_19_adam_hiddel_layer_2_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp.assignvariableop_20_adam_hiddel_layer_2_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_21AssignVariableOp0assignvariableop_21_adam_hiddel_layer_3_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp.assignvariableop_22_adam_hiddel_layer_3_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp,assignvariableop_23_adam_prediction_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_prediction_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp.assignvariableop_25_adam_hiddel_layer_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp,assignvariableop_26_adam_hiddel_layer_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_27AssignVariableOp0assignvariableop_27_adam_hiddel_layer_2_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp.assignvariableop_28_adam_hiddel_layer_2_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_29AssignVariableOp0assignvariableop_29_adam_hiddel_layer_3_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp.assignvariableop_30_adam_hiddel_layer_3_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp,assignvariableop_31_adam_prediction_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp*assignvariableop_32_adam_prediction_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_33AssignVariableOp1assignvariableop_33_adam_hiddel_layer_kernel_vhatIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_34AssignVariableOp/assignvariableop_34_adam_hiddel_layer_bias_vhatIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_35AssignVariableOp3assignvariableop_35_adam_hiddel_layer_2_kernel_vhatIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_36AssignVariableOp1assignvariableop_36_adam_hiddel_layer_2_bias_vhatIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_37AssignVariableOp3assignvariableop_37_adam_hiddel_layer_3_kernel_vhatIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_38AssignVariableOp1assignvariableop_38_adam_hiddel_layer_3_bias_vhatIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_39AssignVariableOp/assignvariableop_39_adam_prediction_kernel_vhatIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp-assignvariableop_40_adam_prediction_bias_vhatIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Õ
Identity_41Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_42IdentityIdentity_41:output:0^NoOp_1*
T0*
_output_shapes
: Â
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_42Identity_42:output:0*g
_input_shapesV
T: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¨

ü
I__inference_hiddel_layer_layer_call_and_return_conditional_losses_2949863

inputs1
matmul_readvariableop_resource:	.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î
À
G__inference_full_model_layer_call_and_return_conditional_losses_2950115	
input'
hiddel_layer_2950094:	#
hiddel_layer_2950096:	*
hiddel_layer_2_2950099:
%
hiddel_layer_2_2950101:	*
hiddel_layer_3_2950104:
%
hiddel_layer_3_2950106:	%
prediction_2950109:	 
prediction_2950111:
identity¢$hiddel_layer/StatefulPartitionedCall¢&hiddel_layer_2/StatefulPartitionedCall¢&hiddel_layer_3/StatefulPartitionedCall¢"prediction/StatefulPartitionedCall
$hiddel_layer/StatefulPartitionedCallStatefulPartitionedCallinputhiddel_layer_2950094hiddel_layer_2950096*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hiddel_layer_layer_call_and_return_conditional_losses_2949863³
&hiddel_layer_2/StatefulPartitionedCallStatefulPartitionedCall-hiddel_layer/StatefulPartitionedCall:output:0hiddel_layer_2_2950099hiddel_layer_2_2950101*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hiddel_layer_2_layer_call_and_return_conditional_losses_2949880µ
&hiddel_layer_3/StatefulPartitionedCallStatefulPartitionedCall/hiddel_layer_2/StatefulPartitionedCall:output:0hiddel_layer_3_2950104hiddel_layer_3_2950106*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hiddel_layer_3_layer_call_and_return_conditional_losses_2949897¤
"prediction/StatefulPartitionedCallStatefulPartitionedCall/hiddel_layer_3/StatefulPartitionedCall:output:0prediction_2950109prediction_2950111*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_prediction_layer_call_and_return_conditional_losses_2949914z
IdentityIdentity+prediction/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿä
NoOpNoOp%^hiddel_layer/StatefulPartitionedCall'^hiddel_layer_2/StatefulPartitionedCall'^hiddel_layer_3/StatefulPartitionedCall#^prediction/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2L
$hiddel_layer/StatefulPartitionedCall$hiddel_layer/StatefulPartitionedCall2P
&hiddel_layer_2/StatefulPartitionedCall&hiddel_layer_2/StatefulPartitionedCall2P
&hiddel_layer_3/StatefulPartitionedCall&hiddel_layer_3/StatefulPartitionedCall2H
"prediction/StatefulPartitionedCall"prediction/StatefulPartitionedCall:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
Ê	
Ã
,__inference_full_model_layer_call_fn_2949940	
input
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_full_model_layer_call_and_return_conditional_losses_2949921o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
Í	
Ä
,__inference_full_model_layer_call_fn_2950142

inputs
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_full_model_layer_call_and_return_conditional_losses_2949921o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§

ù
G__inference_prediction_layer_call_and_return_conditional_losses_2949914

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§

ù
G__inference_prediction_layer_call_and_return_conditional_losses_2950330

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
Á
G__inference_full_model_layer_call_and_return_conditional_losses_2949921

inputs'
hiddel_layer_2949864:	#
hiddel_layer_2949866:	*
hiddel_layer_2_2949881:
%
hiddel_layer_2_2949883:	*
hiddel_layer_3_2949898:
%
hiddel_layer_3_2949900:	%
prediction_2949915:	 
prediction_2949917:
identity¢$hiddel_layer/StatefulPartitionedCall¢&hiddel_layer_2/StatefulPartitionedCall¢&hiddel_layer_3/StatefulPartitionedCall¢"prediction/StatefulPartitionedCall
$hiddel_layer/StatefulPartitionedCallStatefulPartitionedCallinputshiddel_layer_2949864hiddel_layer_2949866*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_hiddel_layer_layer_call_and_return_conditional_losses_2949863³
&hiddel_layer_2/StatefulPartitionedCallStatefulPartitionedCall-hiddel_layer/StatefulPartitionedCall:output:0hiddel_layer_2_2949881hiddel_layer_2_2949883*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hiddel_layer_2_layer_call_and_return_conditional_losses_2949880µ
&hiddel_layer_3/StatefulPartitionedCallStatefulPartitionedCall/hiddel_layer_2/StatefulPartitionedCall:output:0hiddel_layer_3_2949898hiddel_layer_3_2949900*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_hiddel_layer_3_layer_call_and_return_conditional_losses_2949897¤
"prediction/StatefulPartitionedCallStatefulPartitionedCall/hiddel_layer_3/StatefulPartitionedCall:output:0prediction_2949915prediction_2949917*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_prediction_layer_call_and_return_conditional_losses_2949914z
IdentityIdentity+prediction/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿä
NoOpNoOp%^hiddel_layer/StatefulPartitionedCall'^hiddel_layer_2/StatefulPartitionedCall'^hiddel_layer_3/StatefulPartitionedCall#^prediction/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2L
$hiddel_layer/StatefulPartitionedCall$hiddel_layer/StatefulPartitionedCall2P
&hiddel_layer_2/StatefulPartitionedCall&hiddel_layer_2/StatefulPartitionedCall2P
&hiddel_layer_3/StatefulPartitionedCall&hiddel_layer_3/StatefulPartitionedCall2H
"prediction/StatefulPartitionedCall"prediction/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï'

G__inference_full_model_layer_call_and_return_conditional_losses_2950195

inputs>
+hiddel_layer_matmul_readvariableop_resource:	;
,hiddel_layer_biasadd_readvariableop_resource:	A
-hiddel_layer_2_matmul_readvariableop_resource:
=
.hiddel_layer_2_biasadd_readvariableop_resource:	A
-hiddel_layer_3_matmul_readvariableop_resource:
=
.hiddel_layer_3_biasadd_readvariableop_resource:	<
)prediction_matmul_readvariableop_resource:	8
*prediction_biasadd_readvariableop_resource:
identity¢#hiddel_layer/BiasAdd/ReadVariableOp¢"hiddel_layer/MatMul/ReadVariableOp¢%hiddel_layer_2/BiasAdd/ReadVariableOp¢$hiddel_layer_2/MatMul/ReadVariableOp¢%hiddel_layer_3/BiasAdd/ReadVariableOp¢$hiddel_layer_3/MatMul/ReadVariableOp¢!prediction/BiasAdd/ReadVariableOp¢ prediction/MatMul/ReadVariableOp
"hiddel_layer/MatMul/ReadVariableOpReadVariableOp+hiddel_layer_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
hiddel_layer/MatMulMatMulinputs*hiddel_layer/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#hiddel_layer/BiasAdd/ReadVariableOpReadVariableOp,hiddel_layer_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
hiddel_layer/BiasAddBiasAddhiddel_layer/MatMul:product:0+hiddel_layer/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
hiddel_layer/ReluReluhiddel_layer/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$hiddel_layer_2/MatMul/ReadVariableOpReadVariableOp-hiddel_layer_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0¡
hiddel_layer_2/MatMulMatMulhiddel_layer/Relu:activations:0,hiddel_layer_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%hiddel_layer_2/BiasAdd/ReadVariableOpReadVariableOp.hiddel_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¤
hiddel_layer_2/BiasAddBiasAddhiddel_layer_2/MatMul:product:0-hiddel_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
hiddel_layer_2/ReluReluhiddel_layer_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$hiddel_layer_3/MatMul/ReadVariableOpReadVariableOp-hiddel_layer_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0£
hiddel_layer_3/MatMulMatMul!hiddel_layer_2/Relu:activations:0,hiddel_layer_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%hiddel_layer_3/BiasAdd/ReadVariableOpReadVariableOp.hiddel_layer_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¤
hiddel_layer_3/BiasAddBiasAddhiddel_layer_3/MatMul:product:0-hiddel_layer_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
hiddel_layer_3/ReluReluhiddel_layer_3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 prediction/MatMul/ReadVariableOpReadVariableOp)prediction_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
prediction/MatMulMatMul!hiddel_layer_3/Relu:activations:0(prediction/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!prediction/BiasAdd/ReadVariableOpReadVariableOp*prediction_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
prediction/BiasAddBiasAddprediction/MatMul:product:0)prediction/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
prediction/SoftmaxSoftmaxprediction/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
IdentityIdentityprediction/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿö
NoOpNoOp$^hiddel_layer/BiasAdd/ReadVariableOp#^hiddel_layer/MatMul/ReadVariableOp&^hiddel_layer_2/BiasAdd/ReadVariableOp%^hiddel_layer_2/MatMul/ReadVariableOp&^hiddel_layer_3/BiasAdd/ReadVariableOp%^hiddel_layer_3/MatMul/ReadVariableOp"^prediction/BiasAdd/ReadVariableOp!^prediction/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2J
#hiddel_layer/BiasAdd/ReadVariableOp#hiddel_layer/BiasAdd/ReadVariableOp2H
"hiddel_layer/MatMul/ReadVariableOp"hiddel_layer/MatMul/ReadVariableOp2N
%hiddel_layer_2/BiasAdd/ReadVariableOp%hiddel_layer_2/BiasAdd/ReadVariableOp2L
$hiddel_layer_2/MatMul/ReadVariableOp$hiddel_layer_2/MatMul/ReadVariableOp2N
%hiddel_layer_3/BiasAdd/ReadVariableOp%hiddel_layer_3/BiasAdd/ReadVariableOp2L
$hiddel_layer_3/MatMul/ReadVariableOp$hiddel_layer_3/MatMul/ReadVariableOp2F
!prediction/BiasAdd/ReadVariableOp!prediction/BiasAdd/ReadVariableOp2D
 prediction/MatMul/ReadVariableOp prediction/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê	
Ã
,__inference_full_model_layer_call_fn_2950067	
input
unknown:	
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_full_model_layer_call_and_return_conditional_losses_2950027o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
X

 __inference__traced_save_2950476
file_prefix2
.savev2_hiddel_layer_kernel_read_readvariableop0
,savev2_hiddel_layer_bias_read_readvariableop4
0savev2_hiddel_layer_2_kernel_read_readvariableop2
.savev2_hiddel_layer_2_bias_read_readvariableop4
0savev2_hiddel_layer_3_kernel_read_readvariableop2
.savev2_hiddel_layer_3_bias_read_readvariableop0
,savev2_prediction_kernel_read_readvariableop.
*savev2_prediction_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop9
5savev2_adam_hiddel_layer_kernel_m_read_readvariableop7
3savev2_adam_hiddel_layer_bias_m_read_readvariableop;
7savev2_adam_hiddel_layer_2_kernel_m_read_readvariableop9
5savev2_adam_hiddel_layer_2_bias_m_read_readvariableop;
7savev2_adam_hiddel_layer_3_kernel_m_read_readvariableop9
5savev2_adam_hiddel_layer_3_bias_m_read_readvariableop7
3savev2_adam_prediction_kernel_m_read_readvariableop5
1savev2_adam_prediction_bias_m_read_readvariableop9
5savev2_adam_hiddel_layer_kernel_v_read_readvariableop7
3savev2_adam_hiddel_layer_bias_v_read_readvariableop;
7savev2_adam_hiddel_layer_2_kernel_v_read_readvariableop9
5savev2_adam_hiddel_layer_2_bias_v_read_readvariableop;
7savev2_adam_hiddel_layer_3_kernel_v_read_readvariableop9
5savev2_adam_hiddel_layer_3_bias_v_read_readvariableop7
3savev2_adam_prediction_kernel_v_read_readvariableop5
1savev2_adam_prediction_bias_v_read_readvariableop<
8savev2_adam_hiddel_layer_kernel_vhat_read_readvariableop:
6savev2_adam_hiddel_layer_bias_vhat_read_readvariableop>
:savev2_adam_hiddel_layer_2_kernel_vhat_read_readvariableop<
8savev2_adam_hiddel_layer_2_bias_vhat_read_readvariableop>
:savev2_adam_hiddel_layer_3_kernel_vhat_read_readvariableop<
8savev2_adam_hiddel_layer_3_bias_vhat_read_readvariableop:
6savev2_adam_prediction_kernel_vhat_read_readvariableop8
4savev2_adam_prediction_bias_vhat_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ß
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*
valueþBû*B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/vhat/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÁ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B á
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_hiddel_layer_kernel_read_readvariableop,savev2_hiddel_layer_bias_read_readvariableop0savev2_hiddel_layer_2_kernel_read_readvariableop.savev2_hiddel_layer_2_bias_read_readvariableop0savev2_hiddel_layer_3_kernel_read_readvariableop.savev2_hiddel_layer_3_bias_read_readvariableop,savev2_prediction_kernel_read_readvariableop*savev2_prediction_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop5savev2_adam_hiddel_layer_kernel_m_read_readvariableop3savev2_adam_hiddel_layer_bias_m_read_readvariableop7savev2_adam_hiddel_layer_2_kernel_m_read_readvariableop5savev2_adam_hiddel_layer_2_bias_m_read_readvariableop7savev2_adam_hiddel_layer_3_kernel_m_read_readvariableop5savev2_adam_hiddel_layer_3_bias_m_read_readvariableop3savev2_adam_prediction_kernel_m_read_readvariableop1savev2_adam_prediction_bias_m_read_readvariableop5savev2_adam_hiddel_layer_kernel_v_read_readvariableop3savev2_adam_hiddel_layer_bias_v_read_readvariableop7savev2_adam_hiddel_layer_2_kernel_v_read_readvariableop5savev2_adam_hiddel_layer_2_bias_v_read_readvariableop7savev2_adam_hiddel_layer_3_kernel_v_read_readvariableop5savev2_adam_hiddel_layer_3_bias_v_read_readvariableop3savev2_adam_prediction_kernel_v_read_readvariableop1savev2_adam_prediction_bias_v_read_readvariableop8savev2_adam_hiddel_layer_kernel_vhat_read_readvariableop6savev2_adam_hiddel_layer_bias_vhat_read_readvariableop:savev2_adam_hiddel_layer_2_kernel_vhat_read_readvariableop8savev2_adam_hiddel_layer_2_bias_vhat_read_readvariableop:savev2_adam_hiddel_layer_3_kernel_vhat_read_readvariableop8savev2_adam_hiddel_layer_3_bias_vhat_read_readvariableop6savev2_adam_prediction_kernel_vhat_read_readvariableop4savev2_adam_prediction_bias_vhat_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *8
dtypes.
,2*	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Ï
_input_shapes½
º: :	::
::
::	:: : : : : : : : : :	::
::
::	::	::
::
::	::	::
::
::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::% !

_output_shapes
:	: !

_output_shapes
::%"!

_output_shapes
:	:!#

_output_shapes	
::&$"
 
_output_shapes
:
:!%

_output_shapes	
::&&"
 
_output_shapes
:
:!'

_output_shapes	
::%(!

_output_shapes
:	: )

_output_shapes
::*

_output_shapes
: "ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*©
serving_default
7
input.
serving_default_input:0ÿÿÿÿÿÿÿÿÿ>

prediction0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¢c

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses"
_tf_keras_layer
»

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
/iter

0beta_1

1beta_2
	2decay
3learning_ratemYmZm[m\m] m^'m_(m`vavbvcvdve vf'vg(vh
vhati
vhatj
vhatk
vhatl
vhatm
 vhatn
'vhato
(vhatp"
	optimizer
X
0
1
2
3
4
 5
'6
(7"
trackable_list_wrapper
X
0
1
2
3
4
 5
'6
(7"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
þ2û
,__inference_full_model_layer_call_fn_2949940
,__inference_full_model_layer_call_fn_2950142
,__inference_full_model_layer_call_fn_2950163
,__inference_full_model_layer_call_fn_2950067À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ê2ç
G__inference_full_model_layer_call_and_return_conditional_losses_2950195
G__inference_full_model_layer_call_and_return_conditional_losses_2950227
G__inference_full_model_layer_call_and_return_conditional_losses_2950091
G__inference_full_model_layer_call_and_return_conditional_losses_2950115À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ËBÈ
"__inference__wrapped_model_2949845input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
9serving_default"
signature_map
&:$	2hiddel_layer/kernel
 :2hiddel_layer/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ø2Õ
.__inference_hiddel_layer_layer_call_fn_2950259¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_hiddel_layer_layer_call_and_return_conditional_losses_2950270¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
):'
2hiddel_layer_2/kernel
": 2hiddel_layer_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_hiddel_layer_2_layer_call_fn_2950279¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_hiddel_layer_2_layer_call_and_return_conditional_losses_2950290¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
):'
2hiddel_layer_3/kernel
": 2hiddel_layer_3/bias
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
Ú2×
0__inference_hiddel_layer_3_layer_call_fn_2950299¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_hiddel_layer_3_layer_call_and_return_conditional_losses_2950310¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
$:"	2prediction/kernel
:2prediction/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_prediction_layer_call_fn_2950319¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_prediction_layer_call_and_return_conditional_losses_2950330¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÊBÇ
%__inference_signature_wrapper_2950250input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	Ptotal
	Qcount
R	variables
S	keras_api"
_tf_keras_metric
^
	Ttotal
	Ucount
V
_fn_kwargs
W	variables
X	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
P0
Q1"
trackable_list_wrapper
-
R	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
T0
U1"
trackable_list_wrapper
-
W	variables"
_generic_user_object
+:)	2Adam/hiddel_layer/kernel/m
%:#2Adam/hiddel_layer/bias/m
.:,
2Adam/hiddel_layer_2/kernel/m
':%2Adam/hiddel_layer_2/bias/m
.:,
2Adam/hiddel_layer_3/kernel/m
':%2Adam/hiddel_layer_3/bias/m
):'	2Adam/prediction/kernel/m
": 2Adam/prediction/bias/m
+:)	2Adam/hiddel_layer/kernel/v
%:#2Adam/hiddel_layer/bias/v
.:,
2Adam/hiddel_layer_2/kernel/v
':%2Adam/hiddel_layer_2/bias/v
.:,
2Adam/hiddel_layer_3/kernel/v
':%2Adam/hiddel_layer_3/bias/v
):'	2Adam/prediction/kernel/v
": 2Adam/prediction/bias/v
.:,	2Adam/hiddel_layer/kernel/vhat
(:&2Adam/hiddel_layer/bias/vhat
1:/
2Adam/hiddel_layer_2/kernel/vhat
*:(2Adam/hiddel_layer_2/bias/vhat
1:/
2Adam/hiddel_layer_3/kernel/vhat
*:(2Adam/hiddel_layer_3/bias/vhat
,:*	2Adam/prediction/kernel/vhat
%:#2Adam/prediction/bias/vhat
"__inference__wrapped_model_2949845s '(.¢+
$¢!

inputÿÿÿÿÿÿÿÿÿ
ª "7ª4
2

prediction$!

predictionÿÿÿÿÿÿÿÿÿ´
G__inference_full_model_layer_call_and_return_conditional_losses_2950091i '(6¢3
,¢)

inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ´
G__inference_full_model_layer_call_and_return_conditional_losses_2950115i '(6¢3
,¢)

inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 µ
G__inference_full_model_layer_call_and_return_conditional_losses_2950195j '(7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 µ
G__inference_full_model_layer_call_and_return_conditional_losses_2950227j '(7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_full_model_layer_call_fn_2949940\ '(6¢3
,¢)

inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_full_model_layer_call_fn_2950067\ '(6¢3
,¢)

inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_full_model_layer_call_fn_2950142] '(7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_full_model_layer_call_fn_2950163] '(7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ­
K__inference_hiddel_layer_2_layer_call_and_return_conditional_losses_2950290^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_hiddel_layer_2_layer_call_fn_2950279Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ­
K__inference_hiddel_layer_3_layer_call_and_return_conditional_losses_2950310^ 0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_hiddel_layer_3_layer_call_fn_2950299Q 0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿª
I__inference_hiddel_layer_layer_call_and_return_conditional_losses_2950270]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_hiddel_layer_layer_call_fn_2950259P/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¨
G__inference_prediction_layer_call_and_return_conditional_losses_2950330]'(0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_prediction_layer_call_fn_2950319P'(0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¥
%__inference_signature_wrapper_2950250| '(7¢4
¢ 
-ª*
(
input
inputÿÿÿÿÿÿÿÿÿ"7ª4
2

prediction$!

predictionÿÿÿÿÿÿÿÿÿ