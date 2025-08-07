use crate::{converter, CE_OFFSET};
use arrow::array::ArrayData;
use arrow::buffer::{Buffer, NullBuffer};
use arrow::datatypes::ArrowNativeType;
use arrow_array::builder::{
    ArrayBuilder, BinaryBuilder, BooleanBuilder, Date32Builder, Float32Builder, Float64Builder,
    Int32Builder, Int64Builder, ListBuilder, StringBuilder, StructBuilder,
    TimestampNanosecondBuilder, UInt32Builder, UInt64Builder,
};
use arrow_array::types::{
    Date32Type, Float32Type, Float64Type, Int32Type, Int64Type, TimestampNanosecondType,
    UInt32Type, UInt64Type,
};
use arrow_array::{
    Array, ArrayRef, ArrowPrimitiveType, BinaryArray, BooleanArray, Date32Array, Float32Array,
    Float64Array, Int32Array, Int64Array, ListArray, PrimitiveArray, RecordBatch, Scalar,
    StringArray, StructArray, TimestampNanosecondArray, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, TimeUnit};
use chrono::Datelike;
use prost::Message;
use prost_reflect::{DynamicMessage, FieldDescriptor, Kind, MessageDescriptor, Value};
use std::iter::zip;
use std::sync::Arc;

fn field_descriptor_to_data_type(field: &FieldDescriptor) -> DataType {
    let inner_type = match field.kind() {
        Kind::Message(md) => match md.full_name() {
            "google.protobuf.Timestamp" => {
                DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, None)
            }
            "google.type.Date" => DataType::Date32,
            _ => {
                let fields: Vec<Field> = md
                    .fields()
                    .map(|f| {
                        Field::new(
                            f.name(),
                            field_descriptor_to_data_type(&f),
                            f.supports_presence(),
                        )
                    })
                    .collect();
                DataType::Struct(fields.into())
            }
        },
        Kind::Double => DataType::Float64,
        Kind::Float => DataType::Float32,
        Kind::Int32 | Kind::Sint32 | Kind::Sfixed32 => DataType::Int32,
        Kind::Int64 | Kind::Sint64 | Kind::Sfixed64 => DataType::Int64,
        Kind::Uint32 | Kind::Fixed32 => DataType::UInt32,
        Kind::Uint64 | Kind::Fixed64 => DataType::UInt64,
        Kind::Bool => DataType::Boolean,
        Kind::String => DataType::Utf8,
        Kind::Bytes => DataType::Binary,
        Kind::Enum(_) => DataType::Int32,
    };

    if field.is_list() {
        DataType::List(Arc::new(Field::new("item", inner_type, false)))
    } else {
        inner_type
    }
}

fn make_builder(data_type: &DataType, capacity: usize) -> Box<dyn ArrayBuilder> {
    match data_type {
        DataType::Float64 => Box::new(Float64Builder::with_capacity(capacity)),
        DataType::Float32 => Box::new(Float32Builder::with_capacity(capacity)),
        DataType::Int32 => Box::new(Int32Builder::with_capacity(capacity)),
        DataType::Int64 => Box::new(Int64Builder::with_capacity(capacity)),
        DataType::UInt32 => Box::new(UInt32Builder::with_capacity(capacity)),
        DataType::UInt64 => Box::new(UInt64Builder::with_capacity(capacity)),
        DataType::Boolean => Box::new(BooleanBuilder::with_capacity(capacity)),
        DataType::Utf8 => Box::new(StringBuilder::new()),
        DataType::Binary => Box::new(BinaryBuilder::new()),
        DataType::Date32 => Box::new(Date32Builder::with_capacity(capacity)),
        DataType::Timestamp(TimeUnit::Nanosecond, None) => {
            Box::new(TimestampNanosecondBuilder::with_capacity(capacity))
        }
        DataType::List(field) => {
            let values_builder = make_builder(field.data_type(), 0);
            Box::new(ListBuilder::new(values_builder))
        }
        DataType::Struct(fields) => {
            let sub_builders = fields
                .iter()
                .map(|f| make_builder(f.data_type(), capacity))
                .collect();
            Box::new(StructBuilder::new(fields.clone(), sub_builders))
        }
        _ => unimplemented!("Unsupported data type for builder: {:?}", data_type),
    }
}

fn append_singular_value_to_builder(
    builder: &mut dyn ArrayBuilder,
    kind: &Kind,
    value: &Value,
) {
    match kind {
        Kind::Double => builder
            .as_any_mut()
            .downcast_mut::<Float64Builder>()
            .unwrap()
            .append_value(value.as_f64().unwrap()),
        Kind::Float => builder
            .as_any_mut()
            .downcast_mut::<Float32Builder>()
            .unwrap()
            .append_value(value.as_f32().unwrap()),
        Kind::Int32 | Kind::Sint32 | Kind::Sfixed32 => builder
            .as_any_mut()
            .downcast_mut::<Int32Builder>()
            .unwrap()
            .append_value(value.as_i32().unwrap()),
        Kind::Int64 | Kind::Sint64 | Kind::Sfixed64 => builder
            .as_any_mut()
            .downcast_mut::<Int64Builder>()
            .unwrap()
            .append_value(value.as_i64().unwrap()),
        Kind::Uint32 | Kind::Fixed32 => builder
            .as_any_mut()
            .downcast_mut::<UInt32Builder>()
            .unwrap()
            .append_value(value.as_u32().unwrap()),
        Kind::Uint64 | Kind::Fixed64 => builder
            .as_any_mut()
            .downcast_mut::<UInt64Builder>()
            .unwrap()
            .append_value(value.as_u64().unwrap()),
        Kind::Bool => builder
            .as_any_mut()
            .downcast_mut::<BooleanBuilder>()
            .unwrap()
            .append_value(value.as_bool().unwrap()),
        Kind::String => builder
            .as_any_mut()
            .downcast_mut::<StringBuilder>()
            .unwrap()
            .append_value(value.as_str().unwrap()),
        Kind::Bytes => builder
            .as_any_mut()
            .downcast_mut::<BinaryBuilder>()
            .unwrap()
            .append_value(value.as_bytes().unwrap()),
        Kind::Enum(_) => builder
            .as_any_mut()
            .downcast_mut::<Int32Builder>()
            .unwrap()
            .append_value(value.as_enum_number().unwrap()),
        Kind::Message(message_descriptor) => {
            if message_descriptor.full_name() == "google.protobuf.Timestamp" {
                let msg = value.as_message().unwrap();
                let seconds = msg.get_field_by_name("seconds").unwrap().as_i64().unwrap();
                let nanos = msg.get_field_by_name("nanos").unwrap().as_i32().unwrap();
                let total_nanos = seconds * 1_000_000_000 + i64::from(nanos);
                builder
                    .as_any_mut()
                    .downcast_mut::<TimestampNanosecondBuilder>()
                    .unwrap()
                    .append_value(total_nanos);
                return;
            }
            if message_descriptor.full_name() == "google.type.Date" {
                let msg = value.as_message().unwrap();
                let year = msg.get_field_by_name("year").unwrap().as_i32().unwrap();
                let month = msg.get_field_by_name("month").unwrap().as_i32().unwrap();
                let day = msg.get_field_by_name("day").unwrap().as_i32().unwrap();
                if year == 0 && month == 0 && day == 0 {
                    builder
                        .as_any_mut()
                        .downcast_mut::<Date32Builder>()
                        .unwrap()
                        .append_value(0);
                } else {
                    let date =
                        chrono::NaiveDate::from_ymd_opt(year, month as u32, day as u32).unwrap();
                    builder
                        .as_any_mut()
                        .downcast_mut::<Date32Builder>()
                        .unwrap()
                        .append_value(date.num_days_from_ce() - CE_OFFSET);
                }
                return;
            }
            let struct_builder = builder
                .as_any_mut()
                .downcast_mut::<StructBuilder>()
                .unwrap();
            let msg = value.as_message().unwrap();
            for (i, sub_field) in message_descriptor.fields().enumerate() {
                let sub_builder = &mut struct_builder.field_builders_mut()[i];
                if sub_field.supports_presence() && !msg.has_field(&sub_field) {
                    let dt = field_descriptor_to_data_type(&sub_field);
                    append_null_for_builder(sub_builder.as_mut(), &dt);
                } else {
                    let sub_value = msg.get_field(&sub_field);
                    append_value_to_builder(sub_builder.as_mut(), &sub_field, &sub_value);
                }
            }
            struct_builder.append(true);
        }
    }
}

fn append_value_to_builder(
    builder: &mut dyn ArrayBuilder,
    field_descriptor: &FieldDescriptor,
    value: &Value,
) {
    if field_descriptor.is_list() {
        let list_builder = builder
            .as_any_mut()
            .downcast_mut::<ListBuilder<Box<dyn ArrayBuilder>>>()
            .unwrap();
        if let Some(list_values) = value.as_list() {
            for item_value in list_values {
                append_singular_value_to_builder(list_builder.values(), &field_descriptor.kind(), item_value);
            }
        }
        list_builder.append(true);
        return;
    }
    append_singular_value_to_builder(builder, &field_descriptor.kind(), value);
}

fn append_null_for_builder(builder: &mut dyn ArrayBuilder, dt: &DataType) {
    match dt {
        DataType::Float64 => builder
            .as_any_mut()
            .downcast_mut::<Float64Builder>()
            .unwrap()
            .append_null(),
        DataType::Float32 => builder
            .as_any_mut()
            .downcast_mut::<Float32Builder>()
            .unwrap()
            .append_null(),
        DataType::Int32 => builder
            .as_any_mut()
            .downcast_mut::<Int32Builder>()
            .unwrap()
            .append_null(),
        DataType::Int64 => builder
            .as_any_mut()
            .downcast_mut::<Int64Builder>()
            .unwrap()
            .append_null(),
        DataType::UInt32 => builder
            .as_any_mut()
            .downcast_mut::<UInt32Builder>()
            .unwrap()
            .append_null(),
        DataType::UInt64 => builder
            .as_any_mut()
            .downcast_mut::<UInt64Builder>()
            .unwrap()
            .append_null(),
        DataType::Boolean => builder
            .as_any_mut()
            .downcast_mut::<BooleanBuilder>()
            .unwrap()
            .append_null(),
        DataType::Utf8 => builder
            .as_any_mut()
            .downcast_mut::<StringBuilder>()
            .unwrap()
            .append_null(),
        DataType::Binary => builder
            .as_any_mut()
            .downcast_mut::<BinaryBuilder>()
            .unwrap()
            .append_null(),
        DataType::Date32 => builder
            .as_any_mut()
            .downcast_mut::<Date32Builder>()
            .unwrap()
            .append_null(),
        DataType::Timestamp(TimeUnit::Nanosecond, None) => builder
            .as_any_mut()
            .downcast_mut::<TimestampNanosecondBuilder>()
            .unwrap()
            .append_null(),
        DataType::List(_) => builder
            .as_any_mut()
            .downcast_mut::<ListBuilder<Box<dyn ArrayBuilder>>>()
            .unwrap()
            .append_null(),
        DataType::Struct(_) => builder
            .as_any_mut()
            .downcast_mut::<StructBuilder>()
            .unwrap()
            .append_null(),
        _ => unimplemented!("Unsupported data type for append_null: {:?}", dt),
    }
}

fn finish_builder(builder: &mut dyn ArrayBuilder, dt: &DataType) -> ArrayRef {
    match dt {
        DataType::Float64 => Arc::new(builder.as_any_mut().downcast_mut::<Float64Builder>().unwrap().finish()),
        DataType::Float32 => Arc::new(builder.as_any_mut().downcast_mut::<Float32Builder>().unwrap().finish()),
        DataType::Int32 => Arc::new(builder.as_any_mut().downcast_mut::<Int32Builder>().unwrap().finish()),
        DataType::Int64 => Arc::new(builder.as_any_mut().downcast_mut::<Int64Builder>().unwrap().finish()),
        DataType::UInt32 => Arc::new(builder.as_any_mut().downcast_mut::<UInt32Builder>().unwrap().finish()),
        DataType::UInt64 => Arc::new(builder.as_any_mut().downcast_mut::<UInt64Builder>().unwrap().finish()),
        DataType::Boolean => Arc::new(builder.as_any_mut().downcast_mut::<BooleanBuilder>().unwrap().finish()),
        DataType::Utf8 => Arc::new(builder.as_any_mut().downcast_mut::<StringBuilder>().unwrap().finish()),
        DataType::Binary => Arc::new(builder.as_any_mut().downcast_mut::<BinaryBuilder>().unwrap().finish()),
        DataType::Date32 => Arc::new(builder.as_any_mut().downcast_mut::<Date32Builder>().unwrap().finish()),
        DataType::Timestamp(TimeUnit::Nanosecond, None) => Arc::new(builder.as_any_mut().downcast_mut::<TimestampNanosecondBuilder>().unwrap().finish()),
        DataType::List(_) => Arc::new(builder.as_any_mut().downcast_mut::<ListBuilder<Box<dyn ArrayBuilder>>>().unwrap().finish()),
        DataType::Struct(_) => Arc::new(builder.as_any_mut().downcast_mut::<StructBuilder>().unwrap().finish()),
        _ => unimplemented!("Unsupported data type for finish_builder: {:?}", dt),
    }
}

fn is_nullable(field: &FieldDescriptor) -> bool {
    field.supports_presence()
}


pub fn fields_to_arrays(
    messages: &[DynamicMessage],
    message_descriptor: &MessageDescriptor,
) -> Vec<(Arc<Field>, Arc<dyn Array>)> {
    if messages.is_empty() {
        return Vec::new();
    }

    let fields: Vec<FieldDescriptor> = message_descriptor.fields().collect();

    let mut builders: Vec<Box<dyn ArrayBuilder>> = fields
        .iter()
        .map(|field| {
            let data_type = field_descriptor_to_data_type(field);
            make_builder(&data_type, messages.len())
        })
        .collect();

    for message in messages {
        for (i, field) in fields.iter().enumerate() {
            let builder = &mut builders[i];
            if field.supports_presence() && !message.has_field(field) {
                let dt = field_descriptor_to_data_type(field);
                append_null_for_builder(builder.as_mut(), &dt);
            } else {
                let value = message.get_field(field);
                append_value_to_builder(builder.as_mut(), field, &value);
            }
        }
    }

    fields
        .iter()
        .zip(builders.iter_mut())
        .map(|(field, builder)| {
            let dt = field_descriptor_to_data_type(field);
            let arrow_field = Arc::new(Field::new(
                field.name(),
                dt.clone(),
                is_nullable(field),
            ));
            (arrow_field, finish_builder(builder.as_mut(), &dt))
        })
        .collect()
}

fn extract_single_primitive<P: ArrowPrimitiveType>(
    array: &ArrayRef,
    messages: &mut [&mut DynamicMessage],
    field_descriptor: &FieldDescriptor,
    value_creator: &dyn Fn(P::Native) -> Value,
) {
    array
        .as_any()
        .downcast_ref::<PrimitiveArray<P>>()
        .unwrap()
        .iter()
        .enumerate()
        .for_each(|(index, value)| match value {
            None => {}
            Some(x) => {
                let element: &mut DynamicMessage = messages.get_mut(index).unwrap();
                element.set_field(field_descriptor, value_creator(x));
            }
        })
}

fn extract_repeated_primitive_type<P>(
    list_array: &ListArray,
    messages: &mut [&mut DynamicMessage],
    field_descriptor: &FieldDescriptor,
    value_creator: &dyn Fn(P::Native) -> Value,
) where
    P: ArrowPrimitiveType,
{
    let values: &PrimitiveArray<P> = list_array
        .values()
        .as_any()
        .downcast_ref::<PrimitiveArray<P>>()
        .unwrap();

    for (i, message) in messages.iter_mut().enumerate() {
        if !list_array.is_null(i) {
            let start = list_array.value_offsets()[i] as usize;
            let end = list_array.value_offsets()[i + 1] as usize;
            if start < end {
                let slice = values.slice(start, end);
                let values = slice
                    .iter()
                    .map(|value| match value {
                        None => value_creator(P::default_value()),
                        Some(x) => value_creator(x),
                    })
                    .collect();
                message.set_field(field_descriptor, Value::List(values));
            }
        }
    }
}

fn extract_repeated_boolean(
    list_array: &ListArray,
    messages: &mut [&mut DynamicMessage],
    field_descriptor: &FieldDescriptor,
) {
    let values: &BooleanArray = list_array
        .values()
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap();

    for (i, message) in messages.iter_mut().enumerate() {
        if !list_array.is_null(i) {
            let start = list_array.value_offsets()[i] as usize;
            let end = list_array.value_offsets()[i + 1] as usize;
            if start < end {
                let each_values = (start..end)
                    .map(|x| values.value(x))
                    .map(Value::Bool)
                    .collect();

                message.set_field(field_descriptor, Value::List(each_values));
            }
        }
    }
}

pub fn extract_repeated_array(
    array: &ArrayRef,
    field_descriptor: &FieldDescriptor,
    messages: &mut [&mut DynamicMessage],
) {
    let list_array: &ListArray = array.as_any().downcast_ref::<ListArray>().unwrap();
    let values = list_array.values();

    match field_descriptor.kind() {
        Kind::Sfixed32 | Kind::Sint32 | Kind::Int32 => {
            extract_repeated_primitive_type::<Int32Type>(
                list_array,
                messages,
                field_descriptor,
                &Value::I32,
            )
        }
        Kind::Fixed32 | Kind::Uint32 => extract_repeated_primitive_type::<UInt32Type>(
            list_array,
            messages,
            field_descriptor,
            &Value::U32,
        ),
        Kind::Sint64 | Kind::Sfixed64 | Kind::Int64 => {
            extract_repeated_primitive_type::<Int64Type>(
                list_array,
                messages,
                field_descriptor,
                &Value::I64,
            )
        }
        Kind::Fixed64 | Kind::Uint64 => extract_repeated_primitive_type::<UInt64Type>(
            list_array,
            messages,
            field_descriptor,
            &Value::U64,
        ),
        Kind::Float => extract_repeated_primitive_type::<Float32Type>(
            list_array,
            messages,
            field_descriptor,
            &Value::F32,
        ),
        Kind::Double => extract_repeated_primitive_type::<Float64Type>(
            list_array,
            messages,
            field_descriptor,
            &Value::F64,
        ),
        Kind::Bool => extract_repeated_boolean(list_array, messages, field_descriptor),

        Kind::String => {
            let values = values.as_any().downcast_ref::<StringArray>().unwrap();
            for (i, message) in messages.iter_mut().enumerate() {
                if !list_array.is_null(i) {
                    let start = list_array.value_offsets()[i] as usize;
                    let end = list_array.value_offsets()[i + 1] as usize;
                    let values_vec: Vec<Value> = (start..end)
                        .map(|idx| Value::String(values.value(idx).to_string()))
                        .collect();
                    message.set_field(field_descriptor, Value::List(values_vec));
                }
            }
        }
        Kind::Bytes => {
            let values = values.as_any().downcast_ref::<BinaryArray>().unwrap();
            for (i, message) in messages.iter_mut().enumerate() {
                if !list_array.is_null(i) {
                    let start = list_array.value_offsets()[i] as usize;
                    let end = list_array.value_offsets()[i + 1] as usize;
                    let values_vec: Vec<Value> = (start..end)
                        .map(|idx| {
                            Value::Bytes(prost::bytes::Bytes::from(values.value(idx).to_vec()))
                        })
                        .collect();
                    message.set_field(field_descriptor, Value::List(values_vec));
                }
            }
        }
        Kind::Message(_) => {}
        Kind::Enum(_) => {}
    }
}

pub fn extract_singular_array(
    array: &ArrayRef,
    field_descriptor: &FieldDescriptor,
    messages: &mut [&mut DynamicMessage],
) {
    match field_descriptor.kind() {
        Kind::Sfixed32 | Kind::Sint32 | Kind::Int32 => {
            extract_single_primitive::<arrow_array::types::Int32Type>(
                array,
                messages,
                field_descriptor,
                &Value::I32,
            )
        }
        Kind::Fixed32 | Kind::Uint32 => extract_single_primitive::<arrow_array::types::UInt32Type>(
            array,
            messages,
            field_descriptor,
            &Value::U32,
        ),
        Kind::Sfixed64 | Kind::Sint64 | Kind::Int64 => {
            extract_single_primitive::<arrow_array::types::Int64Type>(
                array,
                messages,
                field_descriptor,
                &Value::I64,
            )
        }
        Kind::Fixed64 | Kind::Uint64 => extract_single_primitive::<arrow_array::types::UInt64Type>(
            array,
            messages,
            field_descriptor,
            &Value::U64,
        ),
        Kind::Float => extract_single_primitive::<arrow_array::types::Float32Type>(
            array,
            messages,
            field_descriptor,
            &Value::F32,
        ),
        Kind::Double => extract_single_primitive::<arrow_array::types::Float64Type>(
            array,
            messages,
            field_descriptor,
            &Value::F64,
        ),
        Kind::Bool => {
            // BooleanType doesn't implement primitive type
            extract_single_bool(array, field_descriptor, messages);
        }
        Kind::String => extract_single_string(array, field_descriptor, messages),
        Kind::Bytes => extract_single_bytes(array, field_descriptor, messages),

        Kind::Message(message_descriptor) => {
            extract_single_message(array, field_descriptor, message_descriptor, messages)
        }
        Kind::Enum(_) => {}
    }
}

fn extract_single_string(
    array: &ArrayRef,
    field_descriptor: &FieldDescriptor,
    messages: &mut [&mut DynamicMessage],
) {
    array
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .enumerate()
        .for_each(|(index, value)| match value {
            None => {}
            Some(x) => {
                let element: &mut DynamicMessage = messages.get_mut(index).unwrap();
                element.set_field(field_descriptor, Value::String(x.to_string()));
            }
        })
}

fn extract_single_bytes(
    array: &ArrayRef,
    field_descriptor: &FieldDescriptor,
    messages: &mut [&mut DynamicMessage],
) {
    array
        .as_any()
        .downcast_ref::<BinaryArray>()
        .unwrap()
        .iter()
        .enumerate()
        .for_each(|(index, value)| match value {
            None => {}
            Some(x) => {
                let element: &mut DynamicMessage = messages.get_mut(index).unwrap();

                element.set_field(
                    field_descriptor,
                    Value::Bytes(prost::bytes::Bytes::from(x.to_vec())),
                );
            }
        })
}

fn extract_single_message(
    array: &ArrayRef,
    field_descriptor: &FieldDescriptor,
    message_descriptor: MessageDescriptor,
    messages: &mut [&mut DynamicMessage],
) {
    if message_descriptor.full_name() == "google.protobuf.Timestamp"
        || message_descriptor.full_name() == "google.type.Date"
    {
        // TODO!!!
        return;
    }

    let struct_array = array.as_any().downcast_ref::<StructArray>().unwrap();
    let mut sub_messages: Vec<&mut DynamicMessage> = messages
        .iter_mut()
        .map(|message| {
            message
                .get_field_mut(field_descriptor)
                .as_message_mut()
                .unwrap()
        })
        .collect();

    message_descriptor
        .fields()
        .for_each(|field_descriptor: FieldDescriptor| {
            let column: Option<&ArrayRef> = struct_array.column_by_name(field_descriptor.name());
            match column {
                None => {}
                Some(column) => extract_array(column, &field_descriptor, &mut sub_messages),
            }
        });
    messages.iter_mut().enumerate().for_each(|(i, x)| {
        if !struct_array.is_valid(i) {
            x.clear_field(field_descriptor)
        }
    });
}

fn extract_single_bool(
    array: &ArrayRef,
    field_descriptor: &FieldDescriptor,
    messages: &mut [&mut DynamicMessage],
) {
    array
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap()
        .iter()
        .enumerate()
        .for_each(|(index, value)| match value {
            None => {}
            Some(x) => {
                let element: &mut DynamicMessage = messages.get_mut(index).unwrap();
                element.set_field(field_descriptor, Value::Bool(x));
            }
        })
}

pub fn extract_array(
    array: &ArrayRef,
    field_descriptor: &FieldDescriptor,
    messages: &mut [&mut DynamicMessage],
) {
    if field_descriptor.is_map() {
        // TODO:
    } else if field_descriptor.is_list() {
        extract_repeated_array(array, field_descriptor, messages)
    } else {
        extract_singular_array(array, field_descriptor, messages)
    }
}

pub fn messages_to_record_batch(
    values: &[Vec<u8>],
    message_descriptor: &MessageDescriptor,
) -> RecordBatch {
    let messages: Vec<DynamicMessage> = values
        .iter()
        .map(|x| DynamicMessage::decode(message_descriptor.clone(), x.as_slice()).unwrap())
        .collect();
    let arrays: Vec<(Arc<Field>, Arc<dyn Array>)> =
        converter::fields_to_arrays(&messages, message_descriptor);
    let struct_array = if arrays.is_empty() {
        StructArray::new_empty_fields(messages.len(), None)
    } else {
        StructArray::from(arrays)
    };
    RecordBatch::from(struct_array)
}

pub fn record_batch_to_array(
    record_batch: &RecordBatch,
    message_descriptor: &MessageDescriptor,
) -> ArrayData {
    let mut messages: Vec<DynamicMessage> = (0..record_batch.num_rows())
        .map(|_| DynamicMessage::new(message_descriptor.clone()))
        .collect::<Vec<DynamicMessage>>();
    let mut references: Vec<&mut DynamicMessage> = messages.iter_mut().collect();

    message_descriptor
        .fields()
        .for_each(|field_descriptor: FieldDescriptor| {
            let column: Option<&ArrayRef> = record_batch.column_by_name(field_descriptor.name());
            match column {
                None => {}
                Some(column) => extract_array(column, &field_descriptor, &mut references),
            }
        });
    let mut results = converter::BinaryBuilder::new();

    messages
        .iter()
        .for_each(|x| results.append_value(x.encode_to_vec()));
    results.finish().to_data()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_timestamps() {
        let seconds_field = Arc::new(Field::new("seconds", DataType::Int64, true));
        let nanos_field = Arc::new(Field::new("nanos", DataType::Int32, true));

        //let seconds = vec![1710330693i64, 1710330702i64];
        let seconds_array: Arc<dyn Array> = Arc::new(arrow::array::Int64Array::from(vec![
            1710330693i64,
            1710330702i64,
            0i64,
        ]));
        let nanos_array: Arc<dyn Array> =
            Arc::new(arrow::array::Int32Array::from(vec![1_000, 123_456_789, 0]));

        let arrays = vec![(seconds_field, seconds_array), (nanos_field, nanos_array)];

        let valid = vec![true, true, false];
        let results = convert_timestamps(&arrays, &valid);
        assert_eq!(results.len(), 3);

        let expected: TimestampNanosecondArray = arrow::array::Int64Array::from(vec![
            1710330693i64 * 1_000_000_000i64 + 1_000i64,
            1710330702i64 * 1_000_000_000i64 + 123_456_789i64,
            0,
        ])
        .reinterpret_cast();

        let mask = BooleanArray::from(vec![false, false, true]);
        let expected_with_null = arrow::compute::nullif(&expected, &mask).unwrap();

        assert_eq!(
            results.as_ref().to_data(),
            expected_with_null.as_ref().to_data()
        )
    }

    #[test]
    fn test_convert_timestamps_empty() {
        let seconds_field = Arc::new(Field::new("seconds", DataType::Int64, true));
        let nanos_field = Arc::new(Field::new("nanos", DataType::Int32, true));

        let seconds_array: Arc<dyn Array> =
            Arc::new(arrow::array::Int64Array::from(Vec::<i64>::new()));
        let nanos_array: Arc<dyn Array> =
            Arc::new(arrow::array::Int32Array::from(Vec::<i32>::new()));

        let arrays = vec![(seconds_field, seconds_array), (nanos_field, nanos_array)];
        let valid: Vec<bool> = vec![];
        let results = convert_timestamps(&arrays, &valid);
        assert_eq!(results.len(), 0);

        let expected: TimestampNanosecondArray =
            arrow::array::Int64Array::from(Vec::<i64>::new()).reinterpret_cast();
        assert_eq!(results.as_ref(), &expected)
    }
}
