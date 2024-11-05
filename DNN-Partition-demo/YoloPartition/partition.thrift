service Partition {
    // 이미지를 바이너리 데이터로 전송
    string partition(1: binary image_data, 2: list<i32> labels);

    // 모델을 요청하는 메서드 추가
    binary get_model();  // 서버에서 모델 데이터 요청
}
